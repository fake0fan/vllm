# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import safetensors
from filelock import FileLock

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _get_file_lock(path: str) -> FileLock:
    lock_path = path + ".lock"
    lock = FileLock(lock_path)
    return lock


@dataclass
class ECExampleConnectorMetadata(ECConnectorMetadata):
    def __init__(self):
        self.mm_datas_to_load: list[str] = []
        self.mm_datas_to_save: list[str] = []
        self.mm_datas_to_update: dict[str, int] = {}

    def add_meta_to_load(self, mm_hash: str):
        self.mm_datas_to_load.append(mm_hash)

    def add_meta_to_save(self, mm_hash: str):
        self.mm_datas_to_save.append(mm_hash)

    def add_meta_to_update(self, mm_hash: str):
        if mm_hash not in self.mm_datas_to_update:
            self.mm_datas_to_update[mm_hash] = 0
        self.mm_datas_to_update[mm_hash] += 1


class ECExampleConnector(ECConnectorBase):
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._mm_datas_need_loads: set[str] = set()
        self._mm_datas_need_saves: set[str] = set()
        # list of mm_hash to update meta (read or write depending on role)
        self._mm_datas_need_update_meta: list[str] = []
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is not None:
            self._storage_path = transfer_config.get_from_extra_config(
                "shared_storage_path", "/tmp"
            )
            logger.debug(transfer_config)
            logger.debug("Shared storage path is %s", self._storage_path)
        else:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        # Default deallocate_cache flag
        self._deallocate_cache_enabled = (
            transfer_config.get_from_extra_config("deallocate_cache", False)
            if transfer_config
            else False
        )
        logger.info(
            "deallocate_cache enabled is %s",
            self._deallocate_cache_enabled,
        )

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return
        # Load the EC for each mm data
        for mm_hash in metadata.mm_datas_to_load:
            if mm_hash in encoder_cache:
                continue

            filename = self._generate_filename_debug(mm_hash)
            try:
                ec_cache = safetensors.torch.load_file(filename)["ec_cache"].cuda()
                encoder_cache[mm_hash] = ec_cache
                logger.debug("Success load encoder cache for hash %s", mm_hash)
            except Exception as e:
                logger.error(
                    "Failed to load encoder cache for %s: %s",
                    mm_hash,
                    str(e),
                )

            if self._deallocate_cache_enabled:
                self.update_mm_meta(mm_hash, 1)

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return

        filename = self._generate_filename_debug(mm_hash)
        ec_cache = encoder_cache[mm_hash]
        tensors = {"ec_cache": ec_cache.detach().cpu()}
        safetensors.torch.save_file(tensors, filename)

        if self._deallocate_cache_enabled:
            self.update_mm_meta(mm_hash, 1)

        logger.debug("Save cache successful for mm_hash %s", mm_hash)

    def has_caches(
        self,
        request: "Request",
    ) -> list[bool]:
        """
        Check if cache exist externally for each mm_data of request

        Args:
            request (Request): the request object.

        Returns:
            List of bool indicate that ith mm_data exist in cache or not
        """
        result = []
        for feature in request.mm_features:
            result.append(self._found_match_for_mm_data(feature.identifier))
        return result

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
        local_hit: bool,
        remote_hit: bool,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        if remote_hit and not local_hit:
            self._mm_datas_need_loads.add(mm_hash)
        elif not remote_hit and local_hit:
            self._mm_datas_need_saves.add(mm_hash)
        elif remote_hit and local_hit:
            self._mm_datas_need_update_meta.append(mm_hash)

    def maybe_update_remote_cache_state(self, encoder_cache, **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)

        for mm_hash in metadata.mm_datas_to_save:
            if (not self.is_producer) or (mm_hash not in encoder_cache):
                continue

            self.save_caches(
                encoder_cache=encoder_cache,
                mm_hash=mm_hash,
            )

        for mm_hash, count in metadata.mm_datas_to_update.items():
            self.update_mm_meta(mm_hash, count)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ECExampleConnectorMetadata()
        for mm_hash in self._mm_datas_need_loads:
            meta.add_meta_to_load(mm_hash)
        for mm_hash in self._mm_datas_need_saves:
            meta.add_meta_to_save(mm_hash)
        for mm_hash in self._mm_datas_need_update_meta:
            meta.add_meta_to_update(mm_hash)

        self._mm_datas_need_loads.clear()
        self._mm_datas_need_saves.clear()
        self._mm_datas_need_update_meta.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_mm_data(self, mm_hash) -> bool:
        """Check if the cache is hit for the request."""
        filename = self._generate_filename_debug(mm_hash)
        return os.path.exists(filename)

    def _generate_foldername_debug(
        self,
        mm_hash: str,
        create_folder: bool = True,  # <- now defaults to True
    ) -> str:
        """
        Return the folder in which the cache for this mm_hash lives.
        If `create_folder` is True (default) the directory is created
        recursively the first time it is needed.
        """
        foldername = os.path.join(self._storage_path, mm_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(self, mm_hash: str) -> str:
        """
        Return the full path of the safetensors file for this mm_hash.
        Ensures the parent directory exists because
        `_generate_foldername_debug` is called with its default
        (`create_folder=True`).
        """
        foldername = self._generate_foldername_debug(mm_hash)  # <- folder auto-created
        return os.path.join(foldername, "encoder_cache.safetensors")

    def _generate_meta_filename(self, mm_hash: str) -> str:
        """
        Return the full path of the metadata JSON file for this mm_hash.
        """
        foldername = self._generate_foldername_debug(mm_hash)
        return os.path.join(foldername, "meta.json")

    def update_mm_meta(self, mm_hash: str, count: int) -> None:
        """
        Create or update the metadata file for the given mm_hash.
        Increase read (or write) count by count
        when connector is consumer (or producer).
        When read count matches write count, cache file is removed.
        """
        WRITE_COUNT = "write_count"
        READ_COUNT = "read_count"
        # No-op when deallocation metadata behavior is disabled.
        if not self._deallocate_cache_enabled:
            return

        read_count = count if not self.is_producer else 0
        write_count = count if self.is_producer else 0

        meta_filename = self._generate_meta_filename(mm_hash)

        lock = _get_file_lock(meta_filename)
        # Acquire per-file lock before reading/writing metadata
        with lock:
            if os.path.exists(meta_filename):
                # Update existing meta
                with open(meta_filename, "r+") as f:
                    data = json.load(f)
                    data[WRITE_COUNT] += write_count
                    data[READ_COUNT] += read_count

                if data[WRITE_COUNT] == data[READ_COUNT] and data[READ_COUNT] > 0:
                    tensorfile = self._generate_filename_debug(mm_hash)
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(tensorfile)
                        os.remove(meta_filename)
                    return
            else:
                data = {
                    WRITE_COUNT: write_count,
                    READ_COUNT: read_count,
                }

            with open(meta_filename, "w") as f:
                json.dump(data, f, indent=4)

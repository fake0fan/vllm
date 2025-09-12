# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains a new class `MooncakeStore` that allows developers to
think of EC cache transfer operations as putting new EC cache entries
into a remote ECStore-based lookup buffer and getting existing EC caches
from this remote lookup buffer.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_lookup_buffer.base import (
    ECStoreBufferBase)
from vllm.logger import init_logger

DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB

logger = init_logger(__name__)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    storage_root_dir: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeStoreConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size",
                                           DEFAULT_GLOBAL_SEGMENT_SIZE),
            local_buffer_size=config.get("local_buffer_size",
                                         DEFAULT_LOCAL_BUFFER_SIZE),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
            storage_root_dir=config.get("storage_root_dir", ""),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)


class ECMooncakeStore(ECStoreBufferBase):
    """
    Currently, it only supports zero-copy get/put with
    following data path gpu->cpu->cpu->gpu
    TODO: remove by keys, non-blocking
    """

    def __init__(self, vllm_config: "VllmConfig"):
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.from_file(
                vllm_config.ec_transfer_config.ec_mooncake_config_path
            )
            logger.debug("Mooncake Configuration loaded successfully.")

            # Check if storage_root_dir exists and set environment variable
            if (
                self.config.storage_root_dir is not None
                and self.config.storage_root_dir != ""
            ):
                os.environ["MOONCAKE_STORAGE_ROOT_DIR"] = self.config.storage_root_dir
                logger.info(
                    "Set MOONCAKE_STORAGE_ROOT_DIR to: %s", self.config.storage_root_dir
                )
            
            logger.info("Setting up Mooncake store with parameters:")
            logger.info(f"  local_hostname: {self.config.local_hostname}")
            logger.info(f"  metadata_server: {self.config.metadata_server}")
            logger.info(f"  global_segment_size: {self.config.global_segment_size}")
            logger.info(f"  local_buffer_size: {self.config.local_buffer_size}")
            logger.info(f"  protocol: {self.config.protocol}")
            logger.info(f"  device_name: {self.config.device_name}")
            logger.info(f"  master_server_address: {self.config.master_server_address}")

            self.store.setup(self.config.local_hostname,
                             self.config.metadata_server,
                             self.config.global_segment_size,
                             self.config.local_buffer_size,
                             self.config.protocol, self.config.device_name,
                             self.config.master_server_address)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        # Initialize ReplicateConfig
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1

        logger.info("MooncakeConnector initialized successfully.")

    def close(self):
        self.store.close()
        logger.info("Closed the mooncake store connection")

    def exists(self, keys: List[str]) -> List[bool]:
        return self.store.batch_is_exist(keys)

    def metadata_key(self, key: str) -> str:
        #TODO: no guarantee that there is no (k,v) with this key
        return key + "_metadata"

    def get(self, keys: List[str]) -> List[Optional[torch.Tensor]]:
        """
        Zero-copy batch_get_into
        """
        # Retrieve metadata
        try:
            meta_keys = [self.metadata_key(k) for k in keys]
            meta_exists = self.exists(meta_keys)
            exist_ids = {i for i, exist in enumerate(meta_exists) if exist == 1}
            meta_keys = [meta_key for i, meta_key in enumerate(meta_keys) if i in exist_ids]
            meta_bytes = self.store.get_batch(meta_keys)
        except Exception as e:
            logger.error(f"get_batch for metadata failed: {str(e)}")
            return [None] * len(keys)

        buffers = []
        buffer_ptrs = []
        sizes = []
        for meta_byte in meta_bytes:
            meta_out = json.loads(meta_byte.decode('utf-8'))

            # Retrieve metadata (dtype, shape)
            buffer_dtype = getattr(torch, meta_out['dtype'].split(".")[1])
            buffer_shape = tuple(meta_out['shape'])
            buffer = torch.zeros(buffer_shape, dtype=buffer_dtype)

            # Create and register buffer
            buffers.append(buffer)
            buffer_ptrs.append(buffer.data_ptr())
            sizes.append(buffer.numel() * buffer.element_size())
            self.store.register_buffer(buffer_ptrs[-1], sizes[-1])

        # Fill None first and
        # replace valid keys with corresponding buffers
        results = [None] * len(keys)
        try:
            valid_keys = [key for i, key in enumerate(keys) if i in exist_ids]
            read_bytes = self.store.batch_get_into(valid_keys, buffer_ptrs, sizes)
        except Exception as e:
            logger.error(f"batch_get_into failed: {str(e)}")
        finally:
            # Unregister buffers
            for buffer_ptr in buffer_ptrs:
                self.store.unregister_buffer(buffer_ptr)

        exist_ids = sorted(list(exist_ids))
        for id, buffer, read_byte in zip(exist_ids, buffers, read_bytes):
            if read_byte > 0:
                results[id] = buffer.cuda()

        return results
    
    def put(self, keys: List[str], tensors: List[torch.Tensor]):
        """
        Zero-copy put using put_from
        """
        registered_buffers = []
        try:
            # Prepair metadata
            meta_keys = []
            meta_values = []
            cpu_tensors = []
            for k, tensor in zip(keys, tensors):
                if tensor.get_device() != -1:
                    tensor = tensor.cpu()
                
                assert tensor is not None
                cpu_tensors.append(tensor)
                meta = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
                meta_str = json.dumps(meta)
                meta_bytes = meta_str.encode('utf-8')
                key_meta = self.metadata_key(k)
                meta_keys.append(key_meta)
                meta_values.append(meta_bytes)

            # Register buffers
            buffer_ptrs = []
            buffer_sizes = []
            for tensor in cpu_tensors:
                buffer_ptr = tensor.data_ptr()
                buffer_size = tensor.numel() * tensor.element_size()
                buffer_ptrs.append(buffer_ptr)
                buffer_sizes.append(buffer_size)

                # Zero-copy put
                self.store.register_buffer(buffer_ptr, buffer_size)
                registered_buffers.append(buffer_ptr)

            self.store.batch_put_from(
                keys, buffer_ptrs, buffer_sizes, self.replica_config
            )

            # FIXME: make it more consistent, resilent
            # Only put batch metadata normally (not zero-copy)
            # after putting tensors successfully
            self.store.put_batch(
                meta_keys, meta_values, self.replica_config,
            )
        except Exception as e:
            logger.error(
                f"Failed to put keys {",".join(keys)} using batch_put_from: "
                f"{type(e).__name__}: {str(e)}"
            )
        finally:
            for buffer_ptr in registered_buffers:
                self.store.unregister_buffer(buffer_ptr)

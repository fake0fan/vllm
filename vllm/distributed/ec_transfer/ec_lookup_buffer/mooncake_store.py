# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains a new class `MooncakeStore` that allows developers to
think of EC cache transfer operations as putting new EC cache entries
into a remote ECStore-based lookup buffer and getting existing EC caches
from this remote lookup buffer.
"""

import asyncio
import json
import math
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.utils.tensor_memory_pool import (
    TensorMemoryPool,
)
from vllm.logger import init_logger

METADATA_SUFFIX = "_metadata"
DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB
DEFAULT_TENSOR_POOL_SIZE = 1073741824  # 1.0 GiB

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
    transfer_timeout: int
    replica_num: int
    transfer_buffer_size: int

    @staticmethod
    def from_config(config: dict[str, Any]) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname", "localhost"),
            metadata_server=config.get("metadata_server", ""),
            global_segment_size=config.get(
                "global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE
            ),
            local_buffer_size=config.get(
                "local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE
            ),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address", ""),
            storage_root_dir=config.get("storage_root_dir", ""),
            transfer_timeout=int(config.get("transfer_timeout", 1)),
            replica_num=int(config.get("replica_num", 1)),
            transfer_buffer_size=int(
                config.get("transfer_buffer_size", DEFAULT_TENSOR_POOL_SIZE)
            ),
        )


@dataclass
class MooncakeLoadMeta:
    key: str
    num_token: int

class ECMooncakeStore:
    """
    Currently, it only supports zero-copy get/put with
    following data path gpu->cpu->cpu->gpu
    """

    def __init__(self, vllm_config: "VllmConfig"):
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            if vllm_config.ec_transfer_config is None:
                raise ValueError("ec_transfer_config must be set for ECConnectorBase")

            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.from_config(
                vllm_config.ec_transfer_config.ec_connector_extra_config
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
            logger.info("  local_hostname: %s", self.config.local_hostname)
            logger.info("  metadata_server: %s", self.config.metadata_server)
            logger.info("  global_segment_size: %s", self.config.global_segment_size)
            logger.info("  local_buffer_size: %s", self.config.local_buffer_size)
            logger.info("  protocol: %s", self.config.protocol)
            logger.info("  device_name: %s", self.config.device_name)
            logger.info(
                "  master_server_address: %s", self.config.master_server_address
            )
            logger.info("  transfer_timeout: %s", self.config.transfer_timeout)
            logger.info("  replica_num: %s", self.config.replica_num)
            logger.info(
                "  transfer_buffer_size: %s", self.config.transfer_buffer_size
            )

            self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        # Initialize ReplicateConfig
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = self.config.replica_num

        logger.info("MooncakeConnector initialized successfully.")

        self.tensor_pool = TensorMemoryPool(
            max_block_size=self.config.transfer_buffer_size
        )
        self.pool_lock = threading.Lock()
        self.store.register_buffer(
            self.tensor_pool.base_address, self.config.transfer_buffer_size
        )

        # Put async init
        # queue of unfinished put requests stored by keys
        self.put_queue: set[str] = set()
        self.put_queue_cv = asyncio.Condition()
        self.put_loop = asyncio.new_event_loop()
        self.put_thread = threading.Thread(
            target=self.put_loop.run_forever, daemon=True
        )
        self.put_thread.start()

        max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
        self.io_executor = ThreadPoolExecutor(max_workers=max_workers)

        # model config
        self.embed_size = vllm_config.model_config.get_inputs_embeds_size()
        self.dtype = vllm_config.model_config.dtype \
            if isinstance(vllm_config.model_config.dtype, torch.dtype) \
            else getattr(torch, vllm_config.model_config.dtype)

    def close(self):
        self.wait_for_put()

        if self.put_loop.is_running():
            self.put_loop.call_soon_threadsafe(self.put_loop.stop)
            self.put_thread.join()

        self.put_loop.close()

        self.store.unregister_buffer(
            self.tensor_pool.base_address, self.config.transfer_buffer_size
        )
        self.tensor_pool.cleanup()

        self.store.close()
        logger.info("Closed the mooncake store connection")

    def batch_exists(self, keys: list[str]) -> list[bool]:
        if not keys:
            return []
        return self.store.batch_is_exist(keys)

    def batch_remove(self, keys: list[str]) -> int:
        if not keys:
            return 0
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
        return self.store.remove_by_regex(pattern)

    def metadata_key(self, key: str) -> str:
        # TODO: no guarantee that there is no (k,v) with this key
        return key + METADATA_SUFFIX

    def get(self, key: str) -> torch.Tensor | None:
        logger.error("Single get operation is not supported. Use batch_get instead.")
        raise NotImplementedError(
            "Single get is not supported. Use batch_get([key]) instead."
        )

    def batch_get(self, metas: list[MooncakeLoadMeta], device) -> list[torch.Tensor | None]:
        if not metas:
            return []

        buffer_shapes = []
        buffer_addrs = None
        sizes = []
        for meta in metas:
            buffer_shape = (meta.num_token, self.embed_size)
            element_size = torch.tensor([], dtype=self.dtype).element_size()
            num_elem = math.prod(buffer_shape)
            buffer_size = num_elem * element_size
            sizes.append(buffer_size)
            buffer_shapes.append(buffer_shape)

        with self.pool_lock:
            buffer_addrs = [
                self.tensor_pool.allocate(buffer_size) for buffer_size in sizes
            ]

        # Fill None first and
        # replace valid keys with corresponding buffers
        results = [None] * len(metas)
        try:
            keys = [meta.key for meta in metas]
            read_bytes = self.store.batch_get_into(keys, buffer_addrs, sizes)
        except Exception as e:
            with self.pool_lock:
                self.tensor_pool.batch_free(buffer_addrs)
            logger.error("batch_get_into failed: %s", str(e))
            return results

        for i in range(len(metas)):
            if read_bytes[i] > 0:
                results[i] = self.tensor_pool.load_tensor(
                    buffer_addrs[i], self.dtype, buffer_shapes[i], device
                )
            else:
                logger.debug("fail to load for key %s", metas[i].key)

        with self.pool_lock:
            self.tensor_pool.batch_free(buffer_addrs)

        return results

    def put(self, key: str, tensor: torch.Tensor) -> None:
        logger.error("Single put operation is not supported. Use batch_put instead.")
        raise NotImplementedError(
            "Single put is not supported. Use batch_put([key], [tensor]) instead."
        )

    def wait_for_put(self):
        future = asyncio.run_coroutine_threadsafe(
            self._wait_for_put_async(), self.put_loop
        )
        future.result()  # wait until complete

    async def _wait_for_put_async(self):
        async with self.put_queue_cv:
            while self.put_queue:
                await self.put_queue_cv.wait()

    def batch_put(self, keys: list[str], tensors: list[torch.Tensor]) -> None:
        self.put_loop.call_soon_threadsafe(
            lambda: self.put_loop.create_task(self._batch_put_async(keys, tensors))
        )

    async def _batch_put_async(
        self, keys: list[str], tensors: list[torch.Tensor]
    ) -> None:
        async with self.put_queue_cv:
            self.put_queue.update(keys)

        try:
            await self._zero_copy_batch_put(keys, tensors)
        finally:
            async with self.put_queue_cv:
                self.put_queue.difference_update(keys)
                if not self.put_queue:
                    self.put_queue_cv.notify_all()

    async def _zero_copy_batch_put(
        self, keys: list[str], tensors: list[torch.Tensor]
    ) -> None:
        if not keys:
            return

        # Allocate buffer
        buffer_addrs = []
        buffer_sizes = []
        with self.pool_lock:
            for tensor in tensors:
                buffer_addr = self.tensor_pool.store_tensor(tensor)
                buffer_size = tensor.numel() * tensor.element_size()
                buffer_addrs.append(buffer_addr)
                buffer_sizes.append(buffer_size)

        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.io_executor,
                    self.store.batch_put_from,
                    keys,
                    buffer_addrs,
                    buffer_sizes,
                    self.replica_config,
                ),
                timeout=self.config.transfer_timeout,
            )

            # On success, do not free buffer_addrs
            # Tensor pool will automatically free for us
            buffer_addrs = []
        except asyncio.TimeoutError:
            logger.error(
                "Timeout while putting keys %s (timeout=%s seconds)",
                ",".join(keys),
                self.config.transfer_timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to put keys %s using batch_put_from with error %s",
                ",".join(keys),
                str(e),
            )
        finally:
            if buffer_addrs:
                with self.pool_lock:
                    self.tensor_pool.batch_free(buffer_addrs)

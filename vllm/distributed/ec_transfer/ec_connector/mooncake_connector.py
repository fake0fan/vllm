# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (
    TensorMemoryPool,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.request import Request

MMHash = str
ReqId = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"
CHECK_CACHE_MSG = b"check_cache"

logger = init_logger(__name__)


@dataclass(frozen=True)
class Key:
    mm_hash: MMHash
    req_id: ReqId


class MooncakeECAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    mm_hashes: list[tuple[MMHash, list[ReqId]]]
    remote_mm_addrs: list[int]
    remote_token_bytes: list[int]


@dataclass
class MMHashMeta:
    num_encoder_tokens: int
    mm_addr: int


class MooncakeCacheCheckRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    """Request to check if cache exists for mm_hash."""

    mm_hash: str


class MooncakeCacheCheckResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    """Response indicating cache existence."""

    exists: bool
    num_encoder_tokens: int  # 0 if not exists


@dataclass
class RecvMMHashMeta:
    mm_hash_meta: MMHashMeta
    remote_host: str
    remote_port: int


@dataclass
class FinishedSendMMHashSet:
    set: set[Key]
    lock: threading.Lock


@dataclass
class FinishedReceiveMMHashSet:
    set: set[MMHash]
    finish_recv_cond: asyncio.Condition


class MooncakeECConnectorMetadata(ECConnectorMetadata):
    def __init__(self):
        self.mm_hashes_to_recv: dict[Key, RecvMMHashMeta] = {}
        # mm_hashes whose encoder cache should be saved from HBM to external
        # storage on the producer side.
        self.mm_hashes_to_save: dict[MMHash, MMHashMeta] = {}

    def add_recv_req(
        self,
        req_id: ReqId,
        mm_hash: MMHash,
        mm_hash_meta: MMHashMeta,
        remote_host: str,
        remote_port: int,
    ):
        """Add a request to receive encoder cache from remote."""
        self.mm_hashes_to_recv[Key(mm_hash, req_id)] = RecvMMHashMeta(
            mm_hash_meta=mm_hash_meta,
            remote_host=remote_host,
            remote_port=remote_port,
        )

    def add_save_req(
        self,
        mm_hash: MMHash,
        mm_hash_meta: MMHashMeta,
    ) -> None:
        """Add a request to save encoder cache to external storage."""
        self.mm_hashes_to_save[mm_hash] = mm_hash_meta


class MooncakeECConnector(ECConnectorBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: ECConnectorRole,
    ):
        super().__init__(vllm_config, role)

        assert vllm_config.ec_transfer_config is not None

        if role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeECConnectorScheduler | None = (
                MooncakeECConnectorScheduler(vllm_config)
            )
            self.connector_worker: MooncakeECConnectorWorker | None = None
        elif role == ECConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeECConnectorWorker(vllm_config)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool:
        """Check if encoder cache exists remotely for a single mm item."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_cache_item(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Update state after encoder cache allocation."""
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        encoder_cache_manager=None,
    ) -> ECConnectorMetadata:
        """Build connector metadata for this step."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(
            scheduler_output, encoder_cache_manager
        )

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when request finishes, returns transfer params if needed."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_encoder_cache(
        self,
        transfer_buffer: TensorMemoryPool,
    ):
        """Register encoder cache tensors with Mooncake."""
        assert self.connector_worker is not None
        # For NIXL, we register the main encoder cache tensor
        # Individual mm_hash caches are handled via recv tensors
        if (
            hasattr(self.connector_worker, "transfer_buffer")
            and self.connector_worker.transfer_buffer is not None
        ):
            # Already registered
            return
        # The encoder_cache will be registered when it's first set
        # via register_encoder_cache method
        self.connector_worker.register_encoder_cache(transfer_buffer)

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """Start loading encoder caches from remote via Mooncake."""
        assert self.connector_worker is not None
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeECConnectorMetadata)

        self.connector_worker.start_load_caches(encoder_cache, metadata)

    def wait_for_load(self) -> None:
        assert self.connector_worker is not None
        return self.connector_worker.wait_for_load()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Save encoder cache to remote (handled by request_finished)."""
        assert self.connector_worker is not None
        self.connector_worker.save_caches(encoder_cache, mm_hash)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get finished receiving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def maybe_update_remote_cache_state(self, encoder_cache, **kwargs) -> None:
        """
        Maybe update the remote cache state based on the local encoder cache.

        This method can be used to synchronize or update the state of the
        remote cache based on changes in the local encoder cache.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
        """
        assert self.connector_worker is not None
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeECConnectorMetadata)

        return self.connector_worker.maybe_update_remote_cache_state(
            encoder_cache, metadata
        )


class MooncakeECConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.side_channel_host = get_ip()
        self.side_channel_port = get_mooncake_side_channel_port(vllm_config)

        assert vllm_config.ec_transfer_config is not None
        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer

        # Track mm_hashes that need to be loaded from remote
        self._mm_hashes_need_recv: dict[Key, tuple[Request, int]] = {}

        # ZMQ context for cache check probes (only on consumer side)
        if not self.is_producer:
            self._probe_zmq_ctx = zmq.Context()
            self._cache_check_request_encoder = msgspec.msgpack.Encoder()
            self._cache_check_response_decoder = msgspec.msgpack.Decoder(
                MooncakeCacheCheckResponse
            )
        else:
            self._probe_zmq_ctx = None
            self._cache_check_request_encoder = None
            self._cache_check_response_decoder = None

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool:
        """Check if encoder cache exists remotely for a single mm item.

        Uses real-time probe to query encoder instance directly, avoiding
        reliance on stale ec_transfer_params.
        """
        if self.is_producer:
            # Producer doesn't check remote cache
            return False

        remote_host = self.side_channel_host
        remote_port = self.side_channel_port

        # Probe encoder instance for cache existence
        return self._probe_cache_existence(identifier, remote_host, remote_port)

    def _probe_cache_existence(
        self, mm_hash: str, remote_host: str, remote_port: int
    ) -> bool:
        """Probe encoder instance to check if cache exists.

        Returns True if cache exists, False otherwise.
        """
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()
        path = make_zmq_path("tcp", remote_host, remote_port + tp_rank)

        request = MooncakeCacheCheckRequest(mm_hash=mm_hash)
        request_bytes = self._cache_check_request_encoder.encode(request)

        # Encode message as (msg_type, request)
        msg_bytes = msgspec.msgpack.encode((CHECK_CACHE_MSG, request_bytes))

        try:
            sock = make_zmq_socket(
                self._probe_zmq_ctx, path, zmq.REQ, bind=False, linger=0
            )
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

            sock.send(msg_bytes)
            response_bytes = sock.recv()

            # Decode response
            response = self._cache_check_response_decoder.decode(response_bytes)

            sock.close()

            logger.debug(
                "Cache probe for mm_hash %s: exists=%s", mm_hash, response.exists
            )

            return response.exists

        except zmq.Again:
            logger.warning("Cache probe timeout for mm_hash %s at %s", mm_hash, path)
            return False
        except Exception as e:
            logger.error(
                "Cache probe failed for mm_hash %s at %s: %s", mm_hash, path, e
            )
            return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Update state after encoder cache allocation."""
        ec_transfer_params = getattr(request, "ec_transfer_params", None)
        if not ec_transfer_params:
            return

        mm_hash = request.mm_features[index].identifier

        # ec_transfer_params is now a dict keyed by mm_hash: {mm_hash: {...}}
        # Extract params for this specific mm_hash
        mm_hash_params = ec_transfer_params.get(mm_hash)
        if not mm_hash_params:
            logger.debug(
                "No ec_transfer_params found for mm_hash %s in request %s",
                mm_hash,
                request.request_id,
            )
            return

        if mm_hash_params.get("do_remote_encode"):
            if all(p in mm_hash_params for p in ("remote_host", "remote_port")):
                num_encoder_tokens = request.get_num_encoder_embeds(index)
                self._mm_hashes_need_recv[Key(mm_hash, request.request_id)] = (
                    request,
                    num_encoder_tokens,
                )
                logger.debug("Added mm_hash %s to recv queue", mm_hash)
            else:
                logger.warning(
                    "Got invalid ECTransferParams for mm_hash %s: %s. This "
                    "request will not utilize EC transfer",
                    mm_hash,
                    mm_hash_params,
                )

            # Only trigger 1 EC transfer per mm_hash
            mm_hash_params["do_remote_encode"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        encoder_cache_manager=None,
    ) -> ECConnectorMetadata:
        meta = MooncakeECConnectorMetadata()

        # Convert mm_hashes to metadata
        for key, (request, num_encoder_tokens) in self._mm_hashes_need_recv.items():
            mm_hash = key.mm_hash
            ec_transfer_params = getattr(request, "ec_transfer_params", None)
            if ec_transfer_params:
                mm_hash_params = ec_transfer_params.get(mm_hash)
                if mm_hash_params:
                    meta.add_recv_req(
                        req_id=request.request_id,
                        mm_hash=mm_hash,
                        mm_hash_meta=MMHashMeta(
                            num_encoder_tokens=num_encoder_tokens,
                            mm_addr=0,
                        ),
                        remote_host=mm_hash_params["remote_host"],
                        remote_port=mm_hash_params["remote_port"],
                    )
                else:
                    logger.warning(
                        "No ec_transfer_params found for mm_hash %s in request %s",
                        mm_hash,
                        request.request_id,
                    )

        # Clear the lists once workers start the transfers
        self._mm_hashes_need_recv.clear()

        # 2. Check if any HBM-cached items need to be saved to external storage
        # Only producer needs to save
        if self.is_producer and encoder_cache_manager is not None:
            scheduled_mm_hashes = self._collect_scheduled_mm_hashes(scheduler_output)

            for mm_hash, num_token in scheduled_mm_hashes.items():
                # Skip if already in metadata (from loading)
                if any(k.mm_hash == mm_hash for k in meta.mm_hashes_to_recv):
                    continue

                # Check if external storage doesn't have it but HBM does
                # NOTE: This has_cache_item gate ensures we never save the same
                # mm_hash twice while an external copy still exists in the
                # tensor pool.
                if not self.has_cache_item(mm_hash) and encoder_cache_manager.has_cache(
                    mm_hash
                ):
                    # HBM has but external doesn't - mark for saving
                    meta.add_save_req(
                        mm_hash=mm_hash,
                        mm_hash_meta=MMHashMeta(
                            num_encoder_tokens=num_token,
                            mm_addr=0,
                        ),
                    )
                    logger.debug(
                        "Marking mm_hash %s for saving: HBM has cache but "
                        "external storage doesn't",
                        mm_hash,
                    )

        return meta

    def request_finished(
        self,
        request: "Request",
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if not self.is_producer:
            # Consumer doesn't return params
            return False, None

        # Build params for all mm_hashes in this request
        result_params: dict[MMHash, dict[str, Any]] = {}
        for idx, feature in enumerate(request.mm_features):
            mm_hash = feature.identifier
            # Return params keyed by mm_hash for proxy aggregation
            result_params[mm_hash] = {
                "do_remote_encode": True,
                "remote_host": self.side_channel_host,
                "remote_port": self.side_channel_port,
            }

        return len(result_params) > 0, result_params if result_params else None

    def _collect_scheduled_mm_hashes(
        self, scheduler_output: SchedulerOutput
    ) -> dict[str, int]:
        """
        Collect all mm_hashes from scheduled requests.

        Args:
            scheduler_output: The scheduler output containing scheduled requests

        Returns:
            dict: mm_hash -> num_encoder_tokens mapping
        """
        mm_hashes = {}

        # Collect from scheduled_new_reqs
        for req in scheduler_output.scheduled_new_reqs:
            if hasattr(req, "mm_features") and req.mm_features:
                for feature in req.mm_features:
                    mm_hash = feature.identifier
                    num_tokens = feature.mm_position.get_num_embeds
                    mm_hashes[mm_hash] = num_tokens

        return mm_hashes


class MooncakeECConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        self.engine = TransferEngine()
        self.hostname = get_ip()
        assert vllm_config.ec_transfer_config is not None
        device_name = vllm_config.ec_transfer_config.ec_connector_extra_config.get(
            "device_name"
        )
        ret_value = self.engine.initialize(
            self.hostname, "P2PHANDSHAKE", "rdma", device_name
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        # Mooncake handshake port.
        self.side_channel_port: int = get_mooncake_side_channel_port(vllm_config)

        # Encoder cache registration
        self.dtype = (
            vllm_config.model_config.dtype
            if isinstance(vllm_config.model_config.dtype, torch.dtype)
            else getattr(torch, vllm_config.model_config.dtype)
        )
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        self.embed_size = vllm_config.model_config.get_inputs_embeds_size()
        self.byte_per_token = self.embed_size * dtype_size
        self.device_type = current_platform.device_type

        # stored addr of mm tensor in registered caches (external tensor pool)
        self.local_mm_addrs: dict[MMHash, int] = {}
        # reverse map for pool eviction callback: addr -> mm_hash
        self._addr_to_mm_hash: dict[int, MMHash] = {}
        # Lock protecting both maps above; used by save/send/evict paths.
        self._mm_lock = threading.Lock()

        self.num_workers = int(
            vllm_config.ec_transfer_config.ec_connector_extra_config.get(
                "num_workers", 10
            )
        )
        self.mm_hashes_need_recv: set[MMHash] = set()

        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer

        if self.is_producer:
            # Background thread for sending cache to P.
            self._mooncake_sender_t: threading.Thread | None = None
            # Background thread for processing new sending requests.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="vllm-mooncake-ec-sender",
            )
            logger.debug(
                "Mooncake Encoder: use %d workers to send eccaches", self.num_workers
            )
        else:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=self._receiver_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Prefiller: start receiver thread")

        self.finished_sending_mm_hashes: FinishedSendMMHashSet = FinishedSendMMHashSet(
            set(), threading.Lock()
        )
        self.finished_recving_mm_hashes: FinishedReceiveMMHashSet = (
            FinishedReceiveMMHashSet(set(), asyncio.Condition())
        )

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeECAgentMetadata)
        self._cache_check_decoder = msgspec.msgpack.Decoder(MooncakeCacheCheckRequest)
        self._cache_check_encoder = msgspec.msgpack.Encoder()

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.zmq_ctx.term()
        self.async_zmq_ctx.term()
        if self.is_producer:
            self._sender_executor.shutdown(wait=False)
            if self._mooncake_sender_t:
                self._mooncake_sender_t.join()
        elif self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        addr = self.transfer_buffer.store_tensor(encoder_cache[mm_hash])
        # Update bookkeeping for this mm_hash. We intentionally do this
        # after store_tensor returns to avoid deadlock if the pool evicts
        # internally and calls back into _on_pool_free.
        with self._mm_lock:
            self.local_mm_addrs[mm_hash] = addr
            self._addr_to_mm_hash[addr] = mm_hash

    def _on_pool_free(self, addr: int) -> None:
        """Called by the tensor pool when a block is freed (evict or explicit free)."""
        with self._mm_lock:
            mm_hash = self._addr_to_mm_hash.pop(addr, None)
            if mm_hash is not None:
                self.local_mm_addrs.pop(mm_hash, None)

    def _receiver_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _mooncake_sender(
        self, ready_event: threading.Event, base_port: int, tp_rank: int
    ):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        frontend_path = make_zmq_path("tcp", self.hostname, base_port + tp_rank)
        frontend = make_zmq_socket(self.zmq_ctx, frontend_path, zmq.ROUTER)
        logger.debug("Mooncake sender starting listening on path: %s", frontend_path)

        backend_path = make_zmq_path("inproc", str(uuid.uuid4()))
        backend = make_zmq_socket(self.zmq_ctx, backend_path, zmq.PULL)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        ready_event.set()

        try:
            while True:
                sockets = dict(poller.poll())

                if frontend in sockets:
                    identity, _, msg_bytes = frontend.recv_multipart()
                    # Check message type
                    try:
                        decoded = msgspec.msgpack.decode(msg_bytes)
                        if isinstance(decoded, (list, tuple)) and len(decoded) >= 2:
                            msg_type, request_bytes = decoded[0], decoded[1]
                        else:
                            msg_type = decoded
                            request_bytes = msg_bytes

                        if msg_type == CHECK_CACHE_MSG:
                            # Handle cache check synchronously (fast operation)
                            self._handle_cache_check(identity, request_bytes, frontend)
                        else:
                            # Handle transfer request asynchronously
                            self._sender_executor.submit(
                                self._sender_worker,
                                identity,
                                msg_bytes,
                                backend_path,
                            )
                    except Exception as e:
                        logger.error("Error decoding message: %s", e)
                        # Fall back to treating as transfer request
                        self._sender_executor.submit(
                            self._sender_worker,
                            identity,
                            msg_bytes,
                            backend_path,
                        )

                if backend in sockets:
                    identity, status = backend.recv_multipart()
                    frontend.send_multipart((identity, b"", status))

        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            frontend.close()
            backend.close()

    def _handle_cache_check(
        self, identity: bytes, request_bytes: bytes, frontend: zmq.Socket
    ):
        """Handle cache existence check request synchronously."""
        try:
            # Decode the request
            request = self._cache_check_decoder.decode(request_bytes)
            mm_hash = request.mm_hash

            # Check if cache exists in local_mm_addrs (external tensor pool)
            exists = self.has_cache_item(mm_hash)
            num_encoder_tokens = 0

            if exists:
                # We have the address, but we don't have num_encoder_tokens stored
                # We could store it separately, or return 0 and let consumer
                # figure it out. For now, return 0.
                logger.debug("Cache check: mm_hash %s exists", mm_hash)
            else:
                logger.debug("Cache check: mm_hash %s does not exist", mm_hash)

            response = MooncakeCacheCheckResponse(
                exists=exists,
                num_encoder_tokens=num_encoder_tokens,
            )
            response_bytes = self._cache_check_encoder.encode(response)
            frontend.send_multipart((identity, b"", response_bytes))
        except Exception as e:
            logger.error("Error handling cache check: %s", e)
            # Send error response
            response = MooncakeCacheCheckResponse(exists=False, num_encoder_tokens=0)
            response_bytes = self._cache_check_encoder.encode(response)
            frontend.send_multipart((identity, b"", response_bytes))

    def _sender_worker(
        self, identity: bytes, metadata_bytes: bytes, worker_channel_path: str
    ):
        status = TRANS_ERROR

        try:
            metadata = self._decoder.decode(metadata_bytes)
            self.send_ec_cache(metadata)
            status = TRANS_DONE
        except Exception as e:
            logger.error("Error processing Mooncake handshake: %s", e)
        finally:
            pusher = make_zmq_socket(self.zmq_ctx, worker_channel_path, zmq.PUSH)
            try:
                pusher.send_multipart((identity, status))
            except zmq.ZMQError as e:
                logger.warning(
                    "Internal error, maybe the server is shutting down. Error: %s",
                    e,
                )
            finally:
                pusher.close()

    def send_ec_cache(self, meta: MooncakeECAgentMetadata):
        send_mm_hashes = [mm_hash for (mm_hash, _) in meta.mm_hashes]
        self._send_caches(send_mm_hashes, meta)

        with self.finished_sending_mm_hashes.lock:
            keys: list[Key] = []
            for mm_hash, req_ids in meta.mm_hashes:
                keys.extend([Key(mm_hash, req_id) for req_id in req_ids])
            self.finished_sending_mm_hashes.set.update(keys)

    def _send_caches(
        self,
        send_mm_hashes: list[MMHash],
        agent_meta: MooncakeECAgentMetadata,
    ):
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        remote_mm_addrs = agent_meta.remote_mm_addrs
        remote_token_bytes = agent_meta.remote_token_bytes
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for mm_hash, remote_token_byte, remote_mm_addr in zip(
            send_mm_hashes, remote_token_bytes, remote_mm_addrs
        ):
            if remote_token_byte == 0:
                continue

            with self._mm_lock:
                addr = self.local_mm_addrs.get(mm_hash)
            if addr is None:
                logger.debug(
                    "Skipping send for mm_hash %s: no entry in local_mm_addrs", mm_hash
                )
                continue

            src_ptrs.append(addr)
            dst_ptrs.append(remote_mm_addr)
            lengths.append(remote_token_byte)

            logger.debug(
                "Sending ec_caches for mm_hash %s (%d bytes) to %s",
                mm_hash,
                remote_token_byte,
                remote_session,
            )

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value != 0:
            raise RuntimeError(f"Error in batch_transfer_sync_write: {ret_value}")

        logger.debug(
            "Sending to %s done, took %s",
            remote_session,
            time.perf_counter() - start_time,
        )

    def register_encoder_cache(self, transfer_buffer: TensorMemoryPool):
        """Register the EC Cache data in mooncake."""
        self.transfer_buffer = transfer_buffer
        # Keep local_mm_addrs in sync when the pool evicts (auto_evict or free).
        transfer_buffer.on_free = self._on_pool_free
        ret_value = self.engine.register_memory(
            transfer_buffer.base_address, transfer_buffer.max_block_size
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        logger.debug(
            "registered tensor pool with size=%d", transfer_buffer.max_block_size
        )

        # No need to launch server for consumer node.
        if not self.is_producer:
            return

        ready_event = threading.Event()
        self._mooncake_sender_t = threading.Thread(
            target=self._mooncake_sender,
            args=(ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="ec_mooncake_sender",
        )
        self._mooncake_sender_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_mm_hashes(self) -> set[MMHash]:
        async with self.finished_recving_mm_hashes.finish_recv_cond:
            finished_recving_mm_hashes = self.finished_recving_mm_hashes.set
            self.finished_recving_mm_hashes.set = set()
        return finished_recving_mm_hashes

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        fut = None
        if not self.is_producer:
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_mm_hashes(), self.receiver_loop
            )

        if self.is_producer:
            with self.finished_sending_mm_hashes.lock:
                finished_sending_mm_hashes = set(
                    [key.mm_hash for key in self.finished_sending_mm_hashes.set]
                )
                self.finished_sending_mm_hashes.set = set()
        else:
            finished_sending_mm_hashes = set()

        finished_recving_mm_hashes = fut.result() if fut else set()

        if finished_sending_mm_hashes or finished_recving_mm_hashes:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_mm_hashes),
                len(finished_recving_mm_hashes),
            )

        return finished_sending_mm_hashes or None, finished_recving_mm_hashes or None

    async def receive_ec(
        self,
        path: str,
        mm_hash_items: list[tuple[tuple[MMHash, list[ReqId]], MMHashMeta]],
        ec_cache: dict[str, torch.Tensor],
    ):
        mm_hashes, mm_hashes_meta = map(list, zip(*mm_hash_items))
        metadata = MooncakeECAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            mm_hashes=mm_hashes,
            remote_mm_addrs=[meta.mm_addr for meta in mm_hashes_meta],
            remote_token_bytes=[
                meta.num_encoder_tokens * self.byte_per_token for meta in mm_hashes_meta
            ],
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeAgentMetadata: %d bytes", len(encoded_data)
        )
        logger.debug("Sending ec transfer request for %s on path: %s", mm_hashes, path)

        # Send query for the request.
        sock: zmq.asyncio.Socket = make_zmq_socket(
            self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
        )
        sock.setsockopt(zmq.RCVTIMEO, 60000)
        try:
            await sock.send(encoded_data)
            ret_msg = await sock.recv()
            if ret_msg != TRANS_DONE:
                logger.error(
                    "Error happens during transferring encoder cache for %s, see logs in prefiller.",  # noqa: E501
                    mm_hashes,
                )
                return
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error(
                "MooncakeAgentMetadata transfer failed for %s: %s", mm_hashes, e
            )
            return
        finally:
            sock.close()

        for (mm_hash, _), addr, num_bytes in zip(
            metadata.mm_hashes, metadata.remote_mm_addrs, metadata.remote_token_bytes
        ):
            ec_cache[mm_hash] = self.transfer_buffer.load_tensor(
                addr,
                self.dtype,
                (num_bytes // self.byte_per_token, self.embed_size),
                device=self.device_type,
                copy=True,
            )

        async with self.finished_recving_mm_hashes.finish_recv_cond:
            mm_hashes = [mm_hash for (mm_hash, _) in mm_hashes]
            self.finished_recving_mm_hashes.set.update(mm_hashes)

            if self.finished_recving_mm_hashes.set == self.mm_hashes_need_recv:
                self.finished_recving_mm_hashes.finish_recv_cond.notify_all()

        logger.debug("pulling ec_caches for %s finished", mm_hashes)

    def group_ec_pull(self, metadata: MooncakeECConnectorMetadata):
        ec_pulls: dict[str, dict[MMHash, tuple[list[ReqId], MMHashMeta]]] = defaultdict(
            dict
        )
        for key, meta in metadata.mm_hashes_to_recv.items():
            logger.debug(
                "start_load_ec for request %s from remote engine. "
                "Num of encoder token: %s.",
                key.mm_hash,
                meta.mm_hash_meta.num_encoder_tokens,
            )
            path = make_zmq_path(
                "tcp", meta.remote_host, meta.remote_port + self.tp_rank
            )
            mm_hashes_meta = ec_pulls[path]
            if key.mm_hash not in mm_hashes_meta:
                meta.mm_hash_meta.mm_addr = self.transfer_buffer.allocate(
                    meta.mm_hash_meta.num_encoder_tokens * self.byte_per_token
                )
                mm_hashes_meta[key.mm_hash] = ([key.req_id], meta.mm_hash_meta)
            else:
                req_ids, _ = mm_hashes_meta[key.mm_hash]
                req_ids.append(key.req_id)

        return ec_pulls

    def start_load_caches(
        self, ec_cache: dict[str, torch.Tensor], metadata: MooncakeECConnectorMetadata
    ):
        self.mm_hashes_need_recv = set(
            [key.mm_hash for key in metadata.mm_hashes_to_recv]
        )
        ec_pulls = self.group_ec_pull(metadata)
        for path, mm_hashes_meta in ec_pulls.items():
            mm_hash_items = [
                ((mm_hash, req_ids), meta)
                for mm_hash, (req_ids, meta) in mm_hashes_meta.items()
            ]
            asyncio.run_coroutine_threadsafe(
                self.receive_ec(path, mm_hash_items, ec_cache), self.receiver_loop
            )

    async def _wait_for_load(self) -> None:
        async with self.finished_recving_mm_hashes.finish_recv_cond:
            await self.finished_recving_mm_hashes.finish_recv_cond.wait_for(
                lambda: self.finished_recving_mm_hashes.set == self.mm_hashes_need_recv
            )

    def wait_for_load(self) -> None:
        fut = asyncio.run_coroutine_threadsafe(
            self._wait_for_load(), self.receiver_loop
        )
        fut.result()  # Block until complete

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool:
        """WORKER to check if encoder cache exists remotely for a single mm item."""
        with self._mm_lock:
            return identifier in self.local_mm_addrs

    def maybe_update_remote_cache_state(
        self, encoder_cache, metadata: MooncakeECConnectorMetadata, **kwargs
    ) -> None:
        for mm_hash in metadata.mm_hashes_to_save:
            # make sure is producer, and mm_hash exist in local HBM encoder cache
            if (not self.is_producer) or (mm_hash not in encoder_cache):
                continue

            # Check if external storage doesn't have it but HBM does
            if not self.has_cache_item(mm_hash):
                logger.debug("update_remote_cache_state for hash %s", mm_hash)
                self.save_caches(
                    encoder_cache=encoder_cache,
                    mm_hash=mm_hash,
                )


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_EC_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )

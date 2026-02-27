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
from vllm.distributed.ec_transfer.ec_connector.encoder_cache_transfer_buffer import (
    EncoderCacheTransferBuffer,
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
        # mm_hashes whose encoder cache should be saved from
        # EncodeCacheManager to external storage on the producer side.
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
        request: "Request" = None,
    ) -> bool:
        """Check if encoder cache exists remotely for a single mm item."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_cache_item(identifier, request)

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

    def register_encoder_cache(self, transfer_buffer) -> None:
        """Register encoder cache - no-op as buffer is managed internally.

        This method exists for API compatibility with gpu_model_runner.
        The actual buffer is created and registered in
        MooncakeECConnectorWorker.__init__.
        """
        # Buffer is already initialized in connector_worker.__init__
        pass

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
        request: "Request",
    ) -> bool:
        """Check if encoder cache exists remotely for a single mm item.

        Uses real-time probe to query encoder instance directly, avoiding
        reliance on stale ec_transfer_params.

        Request info (host, port) is a must for consumer to find cache
        """
        logger.info(
            "[EC_SCHEDULER] has_cache_item called for mm_hash=%s, is_producer=%s",
            identifier[:16],
            self.is_producer,
        )

        if self.is_producer:
            # Producer doesn't check remote cache
            logger.debug("[EC_SCHEDULER] Producer node, skipping remote cache check")
            return False

        try:
            ec_transfer_params = getattr(request, "ec_transfer_params", {})
            mm_hash_params = ec_transfer_params.get(identifier, {})
            remote_host = mm_hash_params.get("remote_host")
            remote_port = mm_hash_params.get("remote_port")
            
            if not remote_host or not remote_port:
                logger.error("Missing remote_host or remote_port for hash: %s", identifier)
                return False
        except Exception as e:
            logger.error("Unable to get remote host and port. Error: %s", e)
            return False

        logger.info(
            "[EC_SCHEDULER] Probing encoder at %s:%d for mm_hash=%s",
            remote_host,
            remote_port,
            identifier[:16],
        )

        # Probe encoder instance for cache existence
        result = self._probe_cache_existence(identifier, remote_host, remote_port)
        logger.info(
            "[EC_SCHEDULER] Probe result for mm_hash=%s: exists=%s",
            identifier[:16],
            result,
        )
        return result

    def _probe_cache_existence(
        self, mm_hash: str, remote_host: str, remote_port: int
    ) -> bool:
        """Probe encoder instance to check if cache exists.

        Returns True if cache exists, False otherwise.
        """
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()
        path = make_zmq_path("tcp", remote_host, remote_port + tp_rank)

        logger.info(
            "[EC_PROBE] Starting cache probe: mm_hash=%s, path=%s, tp_rank=%d",
            mm_hash[:16],
            path,
            tp_rank,
        )

        request = MooncakeCacheCheckRequest(mm_hash=mm_hash)
        request_bytes = self._cache_check_request_encoder.encode(request)

        # Encode message as (msg_type, request)
        msg_bytes = msgspec.msgpack.encode((CHECK_CACHE_MSG, request_bytes))

        logger.debug(
            "[EC_PROBE] Sending CHECK_CACHE_MSG: msg_size=%d bytes", len(msg_bytes)
        )

        try:
            sock = make_zmq_socket(
                self._probe_zmq_ctx, path, zmq.REQ, bind=False, linger=0
            )
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

            send_start = time.perf_counter()
            sock.send(msg_bytes)
            logger.debug("[EC_PROBE] Message sent, waiting for response...")

            response_bytes = sock.recv()
            rtt = (time.perf_counter() - send_start) * 1000  # ms

            # Decode response
            response = self._cache_check_response_decoder.decode(response_bytes)

            sock.close()

            logger.info(
                "[EC_PROBE] ✓ Cache probe SUCCESS: mm_hash=%s, exists=%s, "
                "num_tokens=%d, rtt=%.2fms",
                mm_hash[:16],
                response.exists,
                response.num_encoder_tokens,
                rtt,
            )

            return response.exists

        except zmq.Again:
            logger.warning(
                "[EC_PROBE] ✗ Cache probe TIMEOUT (5s) for mm_hash=%s at %s",
                mm_hash[:16],
                path,
            )
            return False
        except Exception as e:
            logger.exception(
                "[EC_PROBE] ✗ Cache probe FAILED for mm_hash=%s at %s: %s",
                mm_hash[:16],
                path,
                e,
            )
            return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Update state after encoder cache allocation."""
        ec_transfer_params = getattr(request, "ec_transfer_params", None)
        if not ec_transfer_params:
            logger.debug(
                "[EC_SCHEDULER] No ec_transfer_params for request %s, skipping",
                request.request_id[:16],
            )
            return

        mm_hash = request.mm_features[index].identifier

        logger.info(
            "[EC_SCHEDULER] update_state_after_alloc: req_id=%s, mm_hash=%s, index=%d",
            request.request_id[:16],
            mm_hash[:16],
            index,
        )

        # ec_transfer_params is now a dict keyed by mm_hash: {mm_hash: {...}}
        # Extract params for this specific mm_hash
        mm_hash_params = ec_transfer_params.get(mm_hash)
        if not mm_hash_params:
            logger.debug(
                "[EC_SCHEDULER] No ec_transfer_params found for mm_hash %s in "
                "request %s",
                mm_hash[:16],
                request.request_id[:16],
            )
            return

        logger.info(
            "[EC_SCHEDULER] ec_transfer_params for mm_hash=%s: do_remote_encode=%s, "
            "remote_host=%s, remote_port=%s",
            mm_hash[:16],
            mm_hash_params.get("do_remote_encode"),
            mm_hash_params.get("remote_host"),
            mm_hash_params.get("remote_port"),
        )

        if mm_hash_params.get("do_remote_encode"):
            if all(p in mm_hash_params for p in ("remote_host", "remote_port")):
                num_encoder_tokens = request.get_num_encoder_embeds(index)
                self._mm_hashes_need_recv[Key(mm_hash, request.request_id)] = (
                    request,
                    num_encoder_tokens,
                )
                logger.info(
                    "[EC_SCHEDULER] ✓ Added to recv queue: mm_hash=%s, req_id=%s, "
                    "num_tokens=%d, remote=%s:%s",
                    mm_hash[:16],
                    request.request_id[:16],
                    num_encoder_tokens,
                    mm_hash_params["remote_host"],
                    mm_hash_params["remote_port"],
                )
            else:
                logger.warning(
                    "[EC_SCHEDULER] ✗ Invalid ECTransferParams for mm_hash %s: %s. "
                    "This request will not utilize EC transfer",
                    mm_hash[:16],
                    mm_hash_params,
                )

            # Only trigger 1 EC transfer per mm_hash
            mm_hash_params["do_remote_encode"] = False
            logger.debug(
                "[EC_SCHEDULER] Set do_remote_encode=False to prevent duplicates"
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        encoder_cache_manager=None,
    ) -> ECConnectorMetadata:
        logger.info(
            "[EC_SCHEDULER] build_connector_meta called: "
            "pending_recv=%d, is_producer=%s",
            len(self._mm_hashes_need_recv),
            self.is_producer,
        )

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
                    logger.info(
                        "[EC_SCHEDULER] Added recv_req to metadata: mm_hash=%s, "
                        "req_id=%s, num_tokens=%d, remote=%s:%s",
                        mm_hash[:16],
                        request.request_id[:16],
                        num_encoder_tokens,
                        mm_hash_params["remote_host"],
                        mm_hash_params["remote_port"],
                    )
                else:
                    logger.warning(
                        "[EC_SCHEDULER] No ec_transfer_params found for mm_hash=%s "
                        "in request=%s",
                        mm_hash[:16],
                        request.request_id[:16],
                    )

        # Clear the lists once workers start the transfers
        num_cleared = len(self._mm_hashes_need_recv)
        self._mm_hashes_need_recv.clear()
        logger.info(
            "[EC_SCHEDULER] Cleared %d items from _mm_hashes_need_recv", num_cleared
        )

        # 2. Save any EncoderCacheManager-cached items to external storage. 
        # Only producer needs to save.
        if self.is_producer and encoder_cache_manager is not None:
            scheduled_mm_hashes = self._collect_scheduled_mm_hashes(scheduler_output)
            logger.info(
                "[EC_SCHEDULER] Producer checking EncoderCacheManager→External sync: "
                "scheduled_mm_hashes=%d",
                len(scheduled_mm_hashes),
            )

            for mm_hash, num_token in scheduled_mm_hashes.items():
                has_hbm = encoder_cache_manager.has_cache(mm_hash)

                logger.debug(
                    "[EC_SCHEDULER] mm_hash=%s: has_hbm=%s",
                    mm_hash[:16],
                    has_hbm,
                )

                if has_hbm:
                    meta.add_save_req(
                        mm_hash=mm_hash,
                        mm_hash_meta=MMHashMeta(
                            num_encoder_tokens=num_token,
                            mm_addr=0,
                        ),
                    )

                    logger.info(
                        "[EC_SCHEDULER] ✓ Marked for saving: mm_hash=%s, num_tokens=%d "
                        "(EncoderCacheManager has cache)",
                        mm_hash[:16],
                        num_token,
                    )

        logger.info(
            "[EC_SCHEDULER] build_connector_meta finished: "
            "mm_hashes_to_recv=%d, mm_hashes_to_save=%d",
            len(meta.mm_hashes_to_recv),
            len(meta.mm_hashes_to_save),
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
        logger.info(
            "[EC_SCHEDULER] request_finished: req_id=%s, is_producer=%s",
            request.request_id[:16],
            self.is_producer,
        )

        if not self.is_producer:
            # Consumer doesn't return params
            logger.debug("[EC_SCHEDULER] Consumer node, no params to return")
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
            logger.info(
                "[EC_SCHEDULER] Returning ec_transfer_params for mm_hash=%s: "
                "host=%s, port=%d",
                mm_hash[:16],
                self.side_channel_host,
                self.side_channel_port,
            )

        logger.info(
            "[EC_SCHEDULER] request_finished returns %d mm_hashes", len(result_params)
        )
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

    # Default buffer size: 1GB
    DEFAULT_BUFFER_SIZE = 1073741824

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

        # Transfer buffer - initialized immediately
        self._buffer_size = int(
            vllm_config.ec_transfer_config.ec_connector_extra_config.get(
                "transfer_buffer_size", self.DEFAULT_BUFFER_SIZE
            )
        )
        self.transfer_buffer: EncoderCacheTransferBuffer = EncoderCacheTransferBuffer(
            buffer_size=self._buffer_size,
            device="cpu",
        )

        # stored addr of mm tensor in registered caches (external tensor pool)
        self.local_mm_addrs: dict[MMHash, int] = {}
        # reverse map for pool eviction callback: addr -> mm_hash
        self._addr_to_mm_hash: dict[int, MMHash] = {}
        # Lock protecting both maps above; used by save/send/evict paths.
        self._mm_lock = threading.Lock()

        # Keep local_mm_addrs in sync when the buffer evicts
        self.transfer_buffer.on_free = self._on_pool_free

        # Register buffer with Mooncake for RDMA
        ret_value = self.engine.register_memory(
            self.transfer_buffer.base_address, self.transfer_buffer.buffer_size
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        logger.info(
            "Initialized and registered EC transfer buffer: size=%d bytes, "
            "base_addr=0x%x",
            self._buffer_size,
            self.transfer_buffer.base_address,
        )

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

        # Launch sender thread for producer node
        if self.is_producer:
            ready_event = threading.Event()
            self._mooncake_sender_t = threading.Thread(
                target=self._mooncake_sender,
                args=(ready_event, self.side_channel_port, self.tp_rank),
                daemon=True,
                name="ec_mooncake_sender",
            )
            self._mooncake_sender_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

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
        logger.info(
            "[EC_WORKER] save_caches called: mm_hash=%s, tensor_shape=%s, tp_rank=%d",
            mm_hash[:16],
            encoder_cache[mm_hash].shape,
            self.tp_rank,
        )

        addr = self.transfer_buffer.store_tensor(encoder_cache[mm_hash])

        logger.info(
            "[EC_WORKER] Tensor stored in pool: mm_hash=%s, addr=0x%x, size=%d bytes",
            mm_hash[:16],
            addr,
            encoder_cache[mm_hash].numel() * encoder_cache[mm_hash].element_size(),
        )

        # Update bookkeeping for this mm_hash. We intentionally do this
        # after store_tensor returns to avoid deadlock if the pool evicts
        # internally and calls back into _on_pool_free.
        with self._mm_lock:
            self.local_mm_addrs[mm_hash] = addr
            self._addr_to_mm_hash[addr] = mm_hash
            logger.debug(
                "[EC_WORKER] Updated bookkeeping: local_mm_addrs[%s]=0x%x, "
                "total_cached=%d",
                mm_hash[:16],
                addr,
                len(self.local_mm_addrs),
            )

    def _on_pool_free(self, addr: int) -> None:
        """Called by the tensor pool when a block is freed (evict or explicit free)."""
        with self._mm_lock:
            mm_hash = self._addr_to_mm_hash.pop(addr, None)
            if mm_hash is not None:
                self.local_mm_addrs.pop(mm_hash, None)
                logger.info(
                    "[EC_WORKER] ♻️  Pool eviction callback: freed mm_hash=%s "
                    "at addr=0x%x, remaining_cached=%d",
                    mm_hash[:16],
                    addr,
                    len(self.local_mm_addrs),
                )
            else:
                logger.debug(
                    "[EC_WORKER] Pool freed unknown addr=0x%x (not in tracking)", addr
                )

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

            logger.info(
                "[EC_WORKER_SENDER] Received CHECK_CACHE_MSG for mm_hash=%s from %s",
                mm_hash[:16],
                identity.hex()[:16],
            )

            # Check if cache exists in local_mm_addrs (external tensor pool)
            exists = self.has_cache_in_buffer(mm_hash)
            num_encoder_tokens = 0

            if exists:
                logger.info(
                    "[EC_WORKER_SENDER] ✓ Cache EXISTS for mm_hash=%s", mm_hash[:16]
                )
            else:
                logger.info(
                    "[EC_WORKER_SENDER] ✗ Cache NOT FOUND for mm_hash=%s", mm_hash[:16]
                )

            response = MooncakeCacheCheckResponse(
                exists=exists,
                num_encoder_tokens=num_encoder_tokens,
            )
            response_bytes = self._cache_check_encoder.encode(response)
            frontend.send_multipart((identity, b"", response_bytes))

            logger.debug(
                "[EC_WORKER_SENDER] Sent CHECK_CACHE response: exists=%s",
                exists,
            )
        except Exception as e:
            logger.exception(
                "[EC_WORKER_SENDER] Error handling cache check: %s",
                e,
            )
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

        logger.info(
            "[EC_WORKER_SENDER] send_ec_cache called: num_mm_hashes=%d, "
            "remote=%s:%d, tp_rank=%d",
            len(send_mm_hashes),
            meta.remote_hostname,
            meta.remote_port,
            self.tp_rank,
        )
        for mm_hash, req_ids in meta.mm_hashes:
            logger.debug(
                "[EC_WORKER_SENDER] Will send mm_hash=%s for req_ids=%s",
                mm_hash[:16],
                [rid[:16] for rid in req_ids],
            )

        self._send_caches(send_mm_hashes, meta)

        with self.finished_sending_mm_hashes.lock:
            keys: list[Key] = []
            for mm_hash, req_ids in meta.mm_hashes:
                keys.extend([Key(mm_hash, req_id) for req_id in req_ids])
            self.finished_sending_mm_hashes.set.update(keys)
            logger.info(
                "[EC_WORKER_SENDER] Marked %d keys as finished sending", len(keys)
            )

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

        logger.info(
            "[EC_WORKER_SENDER] _send_caches preparing batch transfer: "
            "num_items=%d, remote_session=%s",
            len(send_mm_hashes),
            remote_session,
        )

        for mm_hash, remote_token_byte, remote_mm_addr in zip(
            send_mm_hashes, remote_token_bytes, remote_mm_addrs
        ):
            if remote_token_byte == 0:
                logger.debug(
                    "[EC_WORKER_SENDER] Skip mm_hash=%s (remote_token_byte=0)",
                    mm_hash[:16],
                )
                continue

            with self._mm_lock:
                addr = self.local_mm_addrs.get(mm_hash)
            if addr is None:
                logger.warning(
                    "[EC_WORKER_SENDER] ✗ Skipping send for mm_hash=%s: "
                    "no entry in local_mm_addrs",
                    mm_hash[:16],
                )
                continue

            src_ptrs.append(addr)
            dst_ptrs.append(remote_mm_addr)
            lengths.append(remote_token_byte)

            logger.info(
                "[EC_WORKER_SENDER] ✓ Added to batch: mm_hash=%s, "
                "src_addr=0x%x, dst_addr=0x%x, size=%d bytes",
                mm_hash[:16],
                addr,
                remote_mm_addr,
                remote_token_byte,
            )

        if not src_ptrs:
            logger.warning(
                "[EC_WORKER_SENDER] No valid transfers in batch, skipping RDMA"
            )
            return

        logger.info(
            "[EC_WORKER_SENDER] Starting Mooncake RDMA batch_transfer_sync_write: "
            "num_transfers=%d, total_bytes=%d",
            len(src_ptrs),
            sum(lengths),
        )

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        elapsed = time.perf_counter() - start_time

        if ret_value != 0:
            logger.error(
                "[EC_WORKER_SENDER] ✗ batch_transfer_sync_write FAILED: ret=%d",
                ret_value,
            )
            raise RuntimeError(f"Error in batch_transfer_sync_write: {ret_value}")

        bandwidth_gbps = (sum(lengths) / (1024**3)) / elapsed if elapsed > 0 else 0
        logger.info(
            "[EC_WORKER_SENDER] ✓ RDMA transfer SUCCESS: %d transfers, "
            "%.2f MB in %.3fs (%.2f GB/s)",
            len(src_ptrs),
            sum(lengths) / (1024**2),
            elapsed,
            bandwidth_gbps,
        )

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
        encoder_cache: dict[str, torch.Tensor],
    ):
        mm_hashes, mm_hashes_meta = map(list, zip(*mm_hash_items))

        logger.info(
            "[EC_WORKER_RECEIVER] receive_ec called: num_mm_hashes=%d, "
            "path=%s, tp_rank=%d",
            len(mm_hashes),
            path,
            self.tp_rank,
        )

        for (mm_hash, req_ids), meta in mm_hash_items:
            logger.info(
                "[EC_WORKER_RECEIVER] Will receive mm_hash=%s, req_ids=%s, "
                "num_tokens=%d, allocated_addr=0x%x",
                mm_hash[:16],
                [rid[:16] for rid in req_ids],
                meta.num_encoder_tokens,
                meta.mm_addr,
            )

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
        logger.info(
            "[EC_WORKER_RECEIVER] Sending transfer request: "
            "metadata_size=%d bytes, total_bytes=%d",
            len(encoded_data),
            sum(metadata.remote_token_bytes),
        )

        # Send query for the request.
        sock: zmq.asyncio.Socket = make_zmq_socket(
            self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
        )
        sock.setsockopt(zmq.RCVTIMEO, 60000)

        start_time = time.perf_counter()
        try:
            await sock.send(encoded_data)
            logger.debug(
                "[EC_WORKER_RECEIVER] Transfer request sent, waiting for RDMA..."
            )

            ret_msg = await sock.recv()
            elapsed = time.perf_counter() - start_time

            if ret_msg != TRANS_DONE:
                logger.error(
                    "[EC_WORKER_RECEIVER] ✗ Transfer FAILED: got %s instead "
                    "of TRANS_DONE",
                    ret_msg,
                )
                return

            logger.info(
                "[EC_WORKER_RECEIVER] ✓ Transfer acknowledgment received in %.3fs",
                elapsed,
            )
        except zmq.ContextTerminated:
            logger.debug(
                "[EC_WORKER_RECEIVER] ZMQ context terminated, exiting receiver thread."
            )
            return
        except Exception as e:
            logger.exception(
                "[EC_WORKER_RECEIVER] ✗ Transfer request failed: %s",
                e,
            )
            return
        finally:
            sock.close()

        # Load tensors from received buffer
        logger.info(
            "[EC_WORKER_RECEIVER] Loading %d tensors from receive buffer...",
            len(mm_hashes),
        )

        for (mm_hash, _), addr, num_bytes in zip(
            metadata.mm_hashes, metadata.remote_mm_addrs, metadata.remote_token_bytes
        ):
            logger.debug(
                "[EC_WORKER_RECEIVER] Loading mm_hash=%s from addr=0x%x, size=%d bytes",
                mm_hash[:16],
                addr,
                num_bytes,
            )

            encoder_cache[mm_hash] = self.transfer_buffer.load_tensor(
                addr,
                self.dtype,
                (num_bytes // self.byte_per_token, self.embed_size),
                device=self.device_type,
                copy=True,
            )

            logger.info(
                "[EC_WORKER_RECEIVER] ✓ Loaded tensor: mm_hash=%s, shape=%s",
                mm_hash[:16],
                encoder_cache[mm_hash].shape,
            )

        async with self.finished_recving_mm_hashes.finish_recv_cond:
            mm_hashes = [mm_hash for (mm_hash, _) in mm_hashes]
            self.finished_recving_mm_hashes.set.update(mm_hashes)

            logger.info(
                "[EC_WORKER_RECEIVER] Updated finished_recving: %d mm_hashes, "
                "total_finished=%d, need_recv=%d",
                len(mm_hashes),
                len(self.finished_recving_mm_hashes.set),
                len(self.mm_hashes_need_recv),
            )

            if self.finished_recving_mm_hashes.set == self.mm_hashes_need_recv:
                self.finished_recving_mm_hashes.finish_recv_cond.notify_all()
                logger.info(
                    "[EC_WORKER_RECEIVER] ✓ All required mm_hashes received, "
                    "notifying waiters"
                )

        logger.info(
            "[EC_WORKER_RECEIVER] receive_ec finished for %d mm_hashes", len(mm_hashes)
        )

    def group_ec_pull(self, metadata: MooncakeECConnectorMetadata):
        ec_pulls: dict[str, dict[MMHash, tuple[list[ReqId], MMHashMeta]]] = defaultdict(
            dict
        )

        logger.info(
            "[EC_WORKER_RECEIVER] group_ec_pull: grouping %d items by remote path",
            len(metadata.mm_hashes_to_recv),
        )

        for key, meta in metadata.mm_hashes_to_recv.items():
            logger.info(
                "[EC_WORKER_RECEIVER] Processing mm_hash=%s, req_id=%s, "
                "num_tokens=%d, remote=%s:%d",
                key.mm_hash[:16],
                key.req_id[:16],
                meta.mm_hash_meta.num_encoder_tokens,
                meta.remote_host,
                meta.remote_port,
            )

            path = make_zmq_path(
                "tcp", meta.remote_host, meta.remote_port + self.tp_rank
            )
            mm_hashes_meta = ec_pulls[path]

            if key.mm_hash not in mm_hashes_meta:
                # Allocate receive buffer
                alloc_size = meta.mm_hash_meta.num_encoder_tokens * self.byte_per_token
                meta.mm_hash_meta.mm_addr = self.transfer_buffer.allocate(alloc_size)

                logger.info(
                    "[EC_WORKER_RECEIVER] Allocated receive buffer: mm_hash=%s, "
                    "addr=0x%x, size=%d bytes",
                    key.mm_hash[:16],
                    meta.mm_hash_meta.mm_addr,
                    alloc_size,
                )

                mm_hashes_meta[key.mm_hash] = ([key.req_id], meta.mm_hash_meta)
            else:
                req_ids, _ = mm_hashes_meta[key.mm_hash]
                req_ids.append(key.req_id)
                logger.debug(
                    "[EC_WORKER_RECEIVER] mm_hash=%s already in group, "
                    "appending req_id=%s",
                    key.mm_hash[:16],
                    key.req_id[:16],
                )

        logger.info(
            "[EC_WORKER_RECEIVER] group_ec_pull finished: grouped into %d paths, "
            "total_mm_hashes=%d",
            len(ec_pulls),
            sum(len(v) for v in ec_pulls.values()),
        )

        return ec_pulls

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        metadata: MooncakeECConnectorMetadata,
    ):
        self.mm_hashes_need_recv = set(
            [key.mm_hash for key in metadata.mm_hashes_to_recv]
        )

        logger.info(
            "[EC_WORKER_RECEIVER] start_load_caches: need_recv=%d mm_hashes",
            len(self.mm_hashes_need_recv),
        )

        ec_pulls = self.group_ec_pull(metadata)

        logger.info(
            "[EC_WORKER_RECEIVER] Starting %d async receive_ec tasks", len(ec_pulls)
        )

        for path, mm_hashes_meta in ec_pulls.items():
            mm_hash_items = [
                ((mm_hash, req_ids), meta)
                for mm_hash, (req_ids, meta) in mm_hashes_meta.items()
            ]

            logger.info(
                "[EC_WORKER_RECEIVER] Scheduling receive_ec for path=%s, "
                "num_mm_hashes=%d",
                path,
                len(mm_hash_items),
            )

            asyncio.run_coroutine_threadsafe(
                self.receive_ec(path, mm_hash_items, encoder_cache), self.receiver_loop
            )

        logger.info("[EC_WORKER_RECEIVER] All receive_ec tasks scheduled")

    async def _wait_for_load(self) -> None:
        logger.info("[EC_WORKER_RECEIVER] _wait_for_load: waiting for all mm_hashes...")

        async with self.finished_recving_mm_hashes.finish_recv_cond:
            await self.finished_recving_mm_hashes.finish_recv_cond.wait_for(
                lambda: self.finished_recving_mm_hashes.set == self.mm_hashes_need_recv
            )

        logger.info(
            "[EC_WORKER_RECEIVER] ✓ _wait_for_load complete: all mm_hashes received"
        )

    def wait_for_load(self) -> None:
        logger.info("[EC_WORKER_RECEIVER] wait_for_load: blocking until complete...")
        fut = asyncio.run_coroutine_threadsafe(
            self._wait_for_load(), self.receiver_loop
        )
        fut.result()  # Block until complete
        logger.info("[EC_WORKER_RECEIVER] wait_for_load: unblocked")

    def has_cache_in_buffer(
        self,
        identifier: str,
    ) -> bool:
        """Worker to check if encoder cache exists in buffer for a single mm item."""
        with self._mm_lock:
            return identifier in self.local_mm_addrs

    def maybe_update_remote_cache_state(
        self, encoder_cache, metadata: MooncakeECConnectorMetadata, **kwargs
    ) -> None:
        for mm_hash in metadata.mm_hashes_to_save:
            # make sure is producer, and mm_hash exists in local
            # EncodeCacheManager encoder cache
            if (not self.is_producer) or (mm_hash not in encoder_cache):
                continue

            # Check if external storage doesn't have it but HBM does
            if not self.has_cache_in_buffer(mm_hash):
                logger.debug(
                    "update_remote_cache_state for hash %s",
                    mm_hash,
                )
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

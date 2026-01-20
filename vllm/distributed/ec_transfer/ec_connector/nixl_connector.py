# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import logging
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import zmq

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
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.encoder_cache_manager import MemorySegment
    from vllm.v1.request import Request

EngineId = str
MMHash = str

GET_META_MSG = b"get_meta_msg"
GET_MM_ADDR_MSG = b"get_mm_addr"

logger = init_logger(__name__)


# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None

# Supported xPUs and types of encoder cache transfer buffer.
_NIXL_SUPPORTED_XPUS = {
    "cuda": ("cuda",),
    "tpu": ("cpu",),
    "xpu": ("cpu",),
}


class NixlECAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    """Metadata exchanged during NIXL handshake for encoder cache."""
    engine_id: str
    agent_metadata: bytes
    enc_base_addr: int
    enc_token_bytes: int


@dataclass
class ECReqMeta:
    """Metadata for a single encoder cache transfer request."""
    mm_hash: str
    num_encoder_tokens: int
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int


@dataclass
class NixlECConnectorMetadata(ECConnectorMetadata):
    """Metadata passed from scheduler to worker for encoder cache transfers."""

    def __init__(self):
        self.reqs_to_recv: dict[MMHash, ECReqMeta] = {}
        self.reqs_to_send: dict[MMHash, float] = {}

    def add_recv_req(
        self,
        mm_hash: MMHash,
        num_encoder_tokens: int,
        remote_host: str,
        remote_port: int,
        remote_engine_id: str,
        tp_size: int,
    ):
        """Add a request to receive encoder cache from remote."""
        self.reqs_to_recv[mm_hash] = ECReqMeta(
            mm_hash=mm_hash,
            num_encoder_tokens=num_encoder_tokens,
            remote_host=remote_host,
            remote_port=remote_port,
            remote_engine_id=remote_engine_id,
            tp_size=tp_size,
        )


class NixlECConnector(ECConnectorBase):
    """NIXL-based encoder cache connector for disaggregated encoder setups."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        assert vllm_config.ec_transfer_config is not None
        assert vllm_config.ec_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.ec_transfer_config.engine_id

        if role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[NixlECConnectorScheduler] = (
                NixlECConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: Optional[NixlECConnectorWorker] = None
        elif role == ECConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlECConnectorWorker(
                vllm_config, self.engine_id
            )

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_encoder_cache(
        self,
        transfer_buffer: TensorMemoryPool,
    ):
        """Register encoder cache tensors with NIXL."""
        assert self.connector_worker is not None
        if (
            hasattr(self.connector_worker, "transfer_buffer")
            and self.connector_worker.transfer_buffer is not None
        ):
            # Already registered
            return
        self.connector_worker.register_encoder_cache(transfer_buffer)

    def start_load_caches(
        self, encoder_cache, **kwargs
    ) -> None:
        """Start loading encoder caches from remote via NIXL."""
        assert self.connector_worker is not None
        metadata: NixlECConnectorMetadata = self._get_connector_metadata()

        self.connector_worker.start_load_caches(encoder_cache, metadata)

    def wait_for_load(self) -> None:
        """Wait until encoder cache tensors are loaded."""
        assert self.connector_worker is not None
        return self.connector_worker.wait_for_load()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Save encoder cache to remote via NIXL.
        
        Similar to mooncake, we store the tensor in transfer_buffer and
        store the address locally for later use when sending the cache.
        """
        assert self.connector_worker is not None
        if mm_hash in encoder_cache:
            addr = self.connector_worker.transfer_buffer.store_tensor(
                encoder_cache[mm_hash]
            )
            self.connector_worker.local_mm_addrs[mm_hash] = addr
            logger.debug(f"Saving cache for mm_hash {mm_hash} with addr {addr}")


    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get finished receiving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)


    ############################################################
    # Scheduler Side Methods
    ############################################################

    def has_caches(self, request: "Request") -> list[bool]:
        """Check if encoder cache exists remotely for each mm_data."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_caches(request)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Update state after encoder cache allocation."""
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)


    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Build connector metadata for this step."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when request finishes, returns transfer params if needed."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)


class NixlECConnectorScheduler:
    """Scheduler-side implementation of NIXL EC connector."""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_EC_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
        )
        logger.info("Initializing NIXL EC Scheduler %s", engine_id)

        # Track mm_hashes that need to be loaded from remote
        # mm_hash -> (request, num_encoder_tokens)
        self._mm_hashes_need_recv: dict[
            MMHash, tuple["Request", int]
        ] = {}
        # Track mm_hashes that need to be sent (for producer role)
        self._mm_hashes_need_send: dict[MMHash, float] = {}

        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer

        # TODO: find a more elegant way to store & manage mm_base_addr
        self._ENCODER_MM_BASE_ADDRS: dict[EngineId, dict[MMHash, int]] = {}

    def has_caches(self, request: "Request") -> list[bool]:
        """Check if encoder cache exists remotely for each mm_data."""
        result = []

        # Hero brute-force, return all true if is consumer

        # ec_transfer_params = getattr(request, "ec_transfer_params", None)
        # remote_mm_segments = None
        # if ec_transfer_params:
        #     remote_mm_segments = ec_transfer_params.get("remote_mm_segments")

        ec_transfer_params = getattr(request, "ec_transfer_params", None)

        for feature in request.mm_features:
            mm_hash = feature.identifier

            if self.is_producer:
                has_cache = False
            else:
                mm_hash_params = (
                    ec_transfer_params.get(mm_hash) if ec_transfer_params else None
                )
                has_cache = bool(
                    mm_hash_params
                    and mm_hash_params.get("num_encoder_tokens", 0) > 0
                    and all(
                        p in mm_hash_params
                        for p in ("remote_engine_id", "remote_host", "remote_port")
                    )
                )
            result.append(has_cache)
        
        logger.debug(f"has_caches results: {result}")
        return result

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
            if all(
                p in mm_hash_params
                for p in ("remote_engine_id", "remote_host", "remote_port")
            ):
                # Get num_encoder_tokens from the request
                num_encoder_tokens = request.get_num_encoder_tokens(index)
                self._mm_hashes_need_recv[mm_hash] = (
                    request,
                    num_encoder_tokens,
                )
                logger.debug(
                    "Added mm_hash %s to recv queue with num_encoder_tokens: %d",
                    mm_hash,
                    num_encoder_tokens,
                )
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
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Build connector metadata for this step."""
        meta = NixlECConnectorMetadata()

        # Convert mm_hashes to metadata
        for mm_hash, (request, num_encoder_tokens) in self._mm_hashes_need_recv.items():
            ec_transfer_params = getattr(request, "ec_transfer_params", None)
            if ec_transfer_params:
                # Extract params for this specific mm_hash
                mm_hash_params = ec_transfer_params.get(mm_hash)
                logger.debug(f"mm_hash_params for {mm_hash}: {mm_hash_params}")
                if mm_hash_params:
                    meta.add_recv_req(
                        mm_hash=mm_hash,
                        num_encoder_tokens=num_encoder_tokens,
                        remote_host=mm_hash_params["remote_host"],
                        remote_port=mm_hash_params["remote_port"],
                        remote_engine_id=mm_hash_params["remote_engine_id"],
                        tp_size=mm_hash_params.get("tp_size", 1),
                    )
                else:
                    logger.warning(
                        "No ec_transfer_params found for mm_hash %s in request %s",
                        mm_hash,
                        request.request_id,
                    )

        meta.reqs_to_send = self._mm_hashes_need_send

        # Clear the lists once workers start the transfers
        self._mm_hashes_need_recv.clear()
        self._mm_hashes_need_send = {}

        return meta

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when request finishes, returns transfer params if needed.
        
        For encoder instances (producers), returns ec_transfer_params keyed by mm_hash
        in the format: {mm_hash: {do_remote_encode, remote_mm_segments, ...}}
        """
        if not self.is_producer:
            # Consumer doesn't return params
            return False, None

        # Build params for all mm_hashes in this request
        result_params: dict[str, dict[str, Any]] = {}
        for idx, feature in enumerate(request.mm_features):
            mm_hash = feature.identifier
            num_encoder_tokens = request.get_num_encoder_tokens(idx)
            
            # Mark mm_hash to be sent asynchronously
            self._mm_hashes_need_send[mm_hash] = (
                time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
            )

            # Return params keyed by mm_hash for proxy aggregation
            # Format: {mm_hash: {do_remote_encode, num_encoder_tokens, remote_engine_id, ...}}
            # Note: mm_base_addr is not included - consumer will get it directly from encoder_cache
            result_params[mm_hash] = {
                "do_remote_encode": True,
                "num_encoder_tokens": num_encoder_tokens,
                "remote_engine_id": self.engine_id,
                "remote_host": self.side_channel_host,
                "remote_port": self.side_channel_port,
                "tp_size": self.vllm_config.parallel_config.tensor_parallel_size,
            }

        return len(result_params) > 0, result_params if result_params else None


class NixlECConnectorWorker:
    """Worker-side implementation of NIXL EC connector."""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL EC Worker %s", engine_id)

        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()

        # NIXL agent
        self.nixl_wrapper = NixlWrapper(engine_id, None)
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

        # NIXL handshake port
        self.side_channel_port: int = (
            envs.VLLM_NIXL_EC_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
        )

        # Device type and memory registration
        self.device_type = current_platform.device_type
        if self.device_type not in _NIXL_SUPPORTED_XPUS:
            raise RuntimeError(f"{self.device_type} is not supported.")
        if self.device_type == "cuda":
            self.nixl_memory_type = "VRAM"
        elif self.device_type in ("tpu", "xpu"):
            self.nixl_memory_type = "DRAM"
        else:
            raise RuntimeError(
                f"{self.device_type} is not supported for encoder cache transfer."
            )
        self.encoder_cache_dtype: torch.dtype = vllm_config.model_config.dtype

        # Encoder cache registration (using TensorMemoryPool like mooncake)
        self.transfer_buffer: Optional[TensorMemoryPool] = None
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.embed_size = vllm_config.model_config.get_inputs_embeds_size()
    
        # Handle dtype like mooncake does
        self.dtype = (
            vllm_config.model_config.dtype
            if isinstance(vllm_config.model_config.dtype, torch.dtype)
            else getattr(torch, vllm_config.model_config.dtype)
        )
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        self.byte_per_token = self.embed_size * dtype_size
        logger.debug(f"self.hidden_size: {self.hidden_size}, embed_size: {self.embed_size}")
        self._registered_descs: list[Any] = []

        # Remote encoder cache addresses
        self._remote_enc_base_addr: dict[EngineId, tuple[int, int]] = {}
        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}

        # Transfer tracking
        self._recving_metadata: dict[MMHash, ECReqMeta] = {}
        self._recving_transfers: dict[MMHash, list[tuple[int, float]]] = (
            defaultdict(list)
        )
        self._reqs_to_send: dict[MMHash, float] = {}
        self.mm_consumer_counts_by_req: dict[MMHash, int] = defaultdict(int)
        self._mm_hashes_need_recv: set[MMHash] = set()

        # Registered receive tensors and segments
        self._registered_mm_descs: dict[
            str, tuple[int, Any, torch.Tensor, list["MemorySegment"]]
        ] = {}
        self._xfer_side_mm_handle: dict[
            tuple[str, int], tuple[str, int, int]
        ] = {}

        # Encoder cache dict reference on consumer side
        # Passed form the model runner
        self._encoder_cache_dict: dict[str, torch.Tensor] | None = None

        # Stored addresses of mm tensors (similar to mooncake's local_mm_addrs)
        # Used on producer side to look up addresses when sending
        self.local_mm_addrs: dict[MMHash, int] = {}

        # Background handshake handling
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._handshake_initiation_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vllm-nixl-ec-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[MMHash, ECReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        self._handshake_lock = threading.RLock()

        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer

    def __del__(self):
        """Cleanup background threads on destruction."""
        self._handshake_initiation_executor.shutdown(wait=False)
        if self._nixl_handshake_listener_t:
            self._nixl_handshake_listener_t.join(timeout=0)

    def register_encoder_cache(self, transfer_buffer: TensorMemoryPool):
        """Register the EC Cache data in NIXL (similar to mooncake)."""
        self.transfer_buffer = transfer_buffer
        
        # Register memory with NIXL using transfer_buffer's base address and size
        enc_size_bytes = transfer_buffer.max_block_size
        enc_base_addr = transfer_buffer.base_address
        
        caches_data = [(enc_base_addr, enc_size_bytes, 0, "")]
        descs = self.nixl_wrapper.get_reg_descs(
            caches_data, self.nixl_memory_type
        )

        logger.debug(
            "Registering encoder cache descs: base_addr=%d, size=%d",
            enc_base_addr,
            enc_size_bytes,
        )
        self.nixl_wrapper.register_memory(descs)
        logger.debug("Done registering encoder cache descs")
        self._registered_descs.append(descs)

        # Start handshake listener for encoder-only instances
        if self.is_producer:
            # For handshake, we still need enc_token_bytes for compatibility
            # This represents bytes per token in the encoder cache
            enc_token_bytes = self.byte_per_token
            
            metadata = NixlECAgentMetadata(
                engine_id=self.engine_id,
                agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                enc_base_addr=enc_base_addr,
                enc_token_bytes=enc_token_bytes,
            )
            logger.debug(f"Metadata at encoder: {metadata}")
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    metadata,
                    ready_event,
                    self._stop_event,
                    self.side_channel_port,
                ),
                daemon=True,
                name="nixl_ec_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()

    def _get_remote_mm_addr(self, host: str, port: int, mm_hash: str) -> int:
        """Query producer for mm_hash address (similar to mooncake's approach)."""
        path = make_zmq_path("tcp", host, port)
        logger.debug(f"Querying mm_hash address for {mm_hash} on path: {path}")
        
        with zmq_ctx(zmq.REQ, path) as sock:
            msg = msgspec.msgpack.encode((GET_MM_ADDR_MSG, self.tp_rank, mm_hash))
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
            sock.send(msg)
            addr_bytes = sock.recv()
            addr = msgspec.msgpack.decode(addr_bytes)
            logger.debug(f"Received address {addr} for mm_hash {mm_hash}")
            return addr

    def _nixl_handshake_listener(
        self,
        metadata: NixlECAgentMetadata,
        ready_event: threading.Event,
        stop_event: threading.Event,
        port: int,
    ):
        """Background thread for handling NIXL handshake requests."""
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded NixlECAgentMetadata: %s bytes", size_in_bytes)

        # host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        # path = make_zmq_path("tcp", host, base_port + tp_rank)
        # logger.debug("Starting EC listener on path: %s", path)
        # with zmq_ctx(zmq.ROUTER, path) as sock:
        #     ready_event.set()
        #     while True:
        #         identity, _, msg = sock.recv_multipart()
        #         if msg != GET_META_MSG:
        #             logger.warning(
        #                 "EC connection listener got unexpected message %s", msg
        #             )
        #         sock.send_multipart((identity, b"", encoded_data))

        # hero (from kv nixl)
        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                # Decode the message which contains (msg_type, ...)
                decoded = msgspec.msgpack.decode(msg)
                logger.debug(f"hero: decocded msg: {decoded}")

                if isinstance(decoded, list) and len(decoded) >= 2:
                    msg_type, target_tp_rank = decoded[0], decoded[1]
                else:
                    msg_type, target_tp_rank = decoded, 0
                
                logger.debug(
                    "Received message for tp rank %s; msg_type: %s",
                    target_tp_rank,
                    msg_type,
                )
                
                if msg_type == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg_type == GET_MM_ADDR_MSG:
                    # Query for mm_hash address
                    if len(decoded) >= 3:
                        logger.debug(f"decoded: {decoded}")
                        mm_hash = decoded[2]
                        # Get address from local_mm_addrs (set by save_caches)
                        logger.debug(f"hero: self.local_mm_addrs: {self.local_mm_addrs}")
                        addr = self.local_mm_addrs.get(mm_hash, 0)
                        logger.debug(f"hero: decocded addr for mm_hash {mm_hash}: {addr}")
                        sock.send_multipart((identity, b"", msgspec.msgpack.encode(addr)))
                    else:
                        logger.warning("Invalid GET_MM_ADDR_MSG format")
                        sock.send_multipart((identity, b"", msgspec.msgpack.encode(0)))
                else:
                    logger.warning("Connection listener got unexpected message %s", msg_type)
                    sock.send_multipart((identity, b"", encoded_data))

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        """Do a NIXL handshake with a remote encoder instance."""
        start_time = time.perf_counter()

        # tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
        # p_remote_rank = self.tp_rank // tp_ratio
        # path = make_zmq_path("tcp", host, port + p_remote_rank)
        # logger.debug(
        #     "Querying EC metadata on path: %s at remote rank %s",
        #     path,
        #     p_remote_rank,
        # )

        tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
        p_remote_rank = self.tp_rank // tp_ratio

        path = make_zmq_path("tcp", host, port)
        logger.debug(
            "Querying EC metadata on path: %s",
            path,
        )

        with zmq_ctx(zmq.REQ, path) as sock:
            msg = msgspec.msgpack.encode((GET_META_MSG, p_remote_rank))
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
            sock.send(msg)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlECAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            logger.debug(f"Received metadata: {metadata}")
            got_metadata_time = time.perf_counter()
            logger.debug(
                "NIXL EC handshake: get metadata took: %s",
                got_metadata_time - start_time,
            )

            if metadata.engine_id != expected_engine_id:
                raise RuntimeError(
                    f"Remote NIXL EC agent engine ID mismatch. "
                    f"Expected {expected_engine_id}, received {metadata.engine_id}."
                )

            remote_agent_name = self.add_remote_agent(
                metadata, p_remote_rank, remote_tp_size
            )
            setup_agent_time = time.perf_counter()
            logger.debug(
                "NIXL EC handshake: add agent took: %s",
                setup_agent_time - got_metadata_time,
            )

        return {p_remote_rank: remote_agent_name}

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlECAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        """Add remote NIXL agent and prepare descriptors for reading encoder cache."""
        engine_id = nixl_agent_meta.engine_id
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]

        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        else:
            assert self._tp_size[engine_id] == remote_tp_size

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        self._remote_enc_base_addr[engine_id] = (
            nixl_agent_meta.enc_base_addr,
            nixl_agent_meta.enc_token_bytes,
        )
        logger.debug(
            f"Added remote agent {remote_agent_name} for engine {engine_id} "
            f"with enc_base_addr={nixl_agent_meta.enc_base_addr}, "
            f"enc_token_bytes={nixl_agent_meta.enc_token_bytes}"
        )
        return remote_agent_name

    def _background_nixl_handshake(
        self, mm_hash: str, remote_engine_id: EngineId, meta: ECReqMeta
    ):
        """Do NIXL handshake in background and add to _ready_requests when done."""
        fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                meta.remote_host,
                meta.remote_port,
                meta.tp_size,
                remote_engine_id,
            )
            self._handshake_futures[remote_engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("EC handshake with %s failed", eid)

            fut.add_done_callback(done_callback)

        # Check handshake success before proceeding with request
        def request_ready(f: Future[Any], entry=(mm_hash, meta)):
            try:
                # Check if handshake succeeded
                f.result()
                self._ready_requests.put(entry)
                logger.debug(f"Request ready for mm_hash {mm_hash} from engine {remote_engine_id}")
            except Exception:
                # Handshake failed
                logger.exception(
                    "Handshake failed for mm_hash %s", mm_hash
                )

        fut.add_done_callback(request_ready)

    def register_encoder_recv_tensor(
        self,
        mm_hash: str,
        recv_tensor: torch.Tensor,
        local_segments: list["MemorySegment"] | None = None,
    ):
        """Register a receive tensor for encoder cache transfer."""
        base_addr = recv_tensor.data_ptr()
        # size_bytes = recv_tensor.numel() * recv_tensor.element_size()

        # self.device_id = max(recv_tensor.get_device(), 0)
        # caches_data = [(base_addr, size_bytes, self.device_id, "")]

        # descs = self.nixl_wrapper.get_reg_descs(
        #     caches_data, self.nixl_memory_type
        # )
        # logger.debug("Registering descs: %s", caches_data)
        # self.nixl_wrapper.register_memory(descs)
        # logger.debug("Done registering descs")

        # Note: For NIXL, we don't need to register individual mm tensors
        # as they are part of the main encoder cache pool
        descs = None
        self._registered_mm_descs[mm_hash] = (
            base_addr,
            descs,
            recv_tensor,
            local_segments,
        )
        logger.debug(
            f"Registered receive tensor for mm_hash {mm_hash} with base_addr {base_addr}"
        )


    def start_load_caches(
        self,
        encoder_cache,
        metadata: NixlECConnectorMetadata,
    ):
        """Start loading encoder caches from remote via NIXL."""
        assert isinstance(metadata, NixlECConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return

        logger.debug(f"start_load_caches: {len(metadata.reqs_to_recv)} requests to receive")

        # Store encoder cache reference
        self._encoder_cache_dict = encoder_cache

        # Process all receive requests
        for mm_hash, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote_engine_id
            self._recving_metadata[mm_hash] = meta
            logger.debug(
                "start_load_caches for mm_hash %s from remote engine %s",
                mm_hash,
                remote_engine_id,
            )

            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(
                            mm_hash, remote_engine_id, meta
                        )
                        continue

            # Handshake completed, start async read transfer
            self._read_mm_segments(mm_hash, meta, encoder_cache)

        if metadata.reqs_to_recv:   # if not empty
            for mm_hash, meta in metadata.reqs_to_recv.items():
                remote_engine_id = meta.remote_engine_id
                if remote_engine_id not in self._remote_agents:
                    logger.debug(f"hero: {remote_engine_id} not in {self._remote_agents}")
                    logger.debug(f"hero: start wait 2 for request ready before read mm")
                    time.sleep(2)
                    logger.debug(f"hero: waited 2 for request ready before read mm")
                    logger.debug(f"hero: self._remote_agents after wait: {self._remote_agents}")
        logger.debug(f"hero: self._ready_requests.empty(): {self._ready_requests.empty()}")


        # Process requests whose handshakes have finished
        while not self._ready_requests.empty():
            try:
                mm_hash, meta = self._ready_requests.get_nowait()
                self._read_mm_segments(mm_hash, meta, encoder_cache)
            except queue.Empty:
                break

        # Add to requests waiting to be sent
        self._reqs_to_send.update(metadata.reqs_to_send)

        # Track which mm_hashes need to be received
        self._mm_hashes_need_recv = set(metadata.reqs_to_recv.keys())

    def _read_mm_segments(self, mm_hash: str, meta: ECReqMeta, encoder_cache):
        """Read encoder cache from remote via NIXL.
        
        Transfers the entire encoder cache tensor for the given mm_hash.
        """
        remote_engine_id = meta.remote_engine_id
        num_encoder_tokens = meta.num_encoder_tokens

        if num_encoder_tokens == 0:
            # No tokens to read, just send notification
            tp_ratio = (
                self._tp_size[self.engine_id] // self._tp_size[remote_engine_id]
            )
            notif_id = f"{mm_hash}:{tp_ratio}$1".encode()
            agent_name = self._remote_agents[remote_engine_id][0]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return

        # Allocate receive tensor if not already registered
        if mm_hash not in self._registered_mm_descs:
            if remote_engine_id not in self._remote_enc_base_addr:
                logger.error(
                    "Remote encoder base addr for engine %s not found when "
                    "starting transfer for mm_hash %s",
                    remote_engine_id,
                    mm_hash
                )
                return

            # Allocate space in transfer_buffer (similar to mooncake)
            num_bytes = num_encoder_tokens * self.byte_per_token
            local_addr = self.transfer_buffer.allocate(num_bytes)
            
            # Create a view tensor at the allocated address for NIXL transfer
            # We'll load the actual tensor after transfer completes
            shape = (num_encoder_tokens, self.embed_size)
            recv_tensor = self.transfer_buffer.load_tensor(
                local_addr,
                self.dtype,
                shape,
                device=self.device_type,
                copy=False,  # Use view for transfer
            )
            
            encoder_cache[mm_hash] = recv_tensor

            logger.debug(
                "Allocated receive tensor for mm_hash %s with shape %s "
                "(num_tokens=%d, embed_size=%d, addr=%d)",
                mm_hash,
                recv_tensor.shape,
                num_encoder_tokens,
                self.embed_size,
                local_addr,
            )

            self.register_encoder_recv_tensor(mm_hash, recv_tensor)

        base_addr, _ = self._remote_enc_base_addr[remote_engine_id]
        local_base_addr, _, recv_tensor, _ = self._registered_mm_descs[mm_hash]

        tp_ratio = (
            self._tp_size[self.engine_id] // self._tp_size[remote_engine_id]
        )
        notif_id = f"{mm_hash}:{tp_ratio}$1".encode()
        agent_name = self._remote_agents[remote_engine_id][0]

        # Transfer the whole tensor: offset 0, length num_encoder_tokens
        # Use byte_per_token for consistency (same model should have same embed_size)
        seg_bytes = num_encoder_tokens * self.byte_per_token
        
        # Get remote address by querying the producer (similar to mooncake's approach)
        # The producer stores addresses in local_mm_addrs when save_caches is called
        remote_addr = self._get_remote_mm_addr(
            meta.remote_host, meta.remote_port, mm_hash
        )
        if remote_addr == 0:
            logger.error(
                "Failed to get remote address for mm_hash %s from producer %s:%s",
                mm_hash, meta.remote_host, meta.remote_port
            )
            return
        
        local_addr = local_base_addr  # Start from local base address (offset 0)

        remote_segments = [(remote_addr, seg_bytes, 0)]
        local_seg_addrs = [(local_addr, seg_bytes, 0)]
        idx = [0]  # Single segment

        logger.debug(
            f"Transferring mm_hash {mm_hash}: remote_addr={remote_addr}, "
            f"local_addr={local_addr}, num_encoder_tokens={num_encoder_tokens}"
        )

        src_descs = self.nixl_wrapper.get_xfer_descs(
            local_seg_addrs, self.nixl_memory_type
        )
        src_xfer_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", src_descs
        )

        dst_descs = self.nixl_wrapper.get_xfer_descs(
            remote_segments, self.nixl_memory_type
        )
        dst_xfer_handle = self.nixl_wrapper.prep_xfer_dlist(
            agent_name, dst_descs
        )

        handle = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            src_xfer_handle,
            idx,
            dst_xfer_handle,
            idx,
            notif_msg=notif_id,
        )

        self.nixl_wrapper.transfer(handle)

        self._xfer_side_mm_handle[(mm_hash, handle)] = (
            mm_hash,
            src_xfer_handle,
            dst_xfer_handle,
        )

        self._recving_transfers[mm_hash].append((handle, time.perf_counter()))

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get finished receiving and sending requests."""
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)

        # Copy received data to encoder cache if needed
        for mm_hash in done_recving:
            logger.debug(f"mm_hash {mm_hash} done recving")
            if mm_hash in self._recving_metadata:
                meta = self._recving_metadata.pop(mm_hash)
                self._copy_recv_to_encoder_cache(mm_hash)

        # Handle timeout - use while loop to process all expired caches
        now = time.perf_counter()
        expired_mm_hashes = []
        # Process all expired requests in a single pass
        for mm_hash, expires in list(self._reqs_to_send.items()):
            if now >= expires:
                count = self.mm_consumer_counts_by_req.pop(mm_hash, 0)
                logger.warning(
                    "Releasing expired EC cache for mm_hash %s which was "
                    "retrieved by %d consumer(s) within %d seconds.",
                    mm_hash,
                    count,
                    envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
                )
                expired_mm_hashes.append(mm_hash)
                done_sending.add(mm_hash)
        
        # Clean up expired entries
        while expired_mm_hashes:
            mm_hash = expired_mm_hashes.pop()
            self._reqs_to_send.pop(mm_hash, None)

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
            )

        return done_sending if done_sending else None, (
            done_recving if done_recving else None
        )

    def _copy_recv_to_encoder_cache(self, mm_hash: str):
        """Mark received encoder cache tensor as ready."""
        if mm_hash not in self._registered_mm_descs:
            logger.warning(
                f"mm_hash {mm_hash} not in registered descriptors, skipping copy"
            )
            return

        local_addr, _, recv_tensor_view, _ = self._registered_mm_descs[mm_hash]
        
        # Load tensor from transfer_buffer with copy=True (similar to mooncake)
        # The view tensor was used for transfer, now load a proper copy
        shape = recv_tensor_view.shape
        loaded_tensor = self.transfer_buffer.load_tensor(
            local_addr,
            self.dtype,
            shape,
            device=self.device_type,
            copy=True,
        )

        # Copy the loaded tensor to encoder cache dict
        if self._encoder_cache_dict is not None:
            self._encoder_cache_dict[mm_hash] = loaded_tensor
            logger.debug(
                "Loaded encoder cache for mm_hash %s, shape: %s",
                mm_hash,
                loaded_tensor.shape,
            )

        logger.debug(
            "Encoder cache transfer completed for mm_hash %s, shape: %s",
            mm_hash,
            loaded_tensor.shape,
        )

    def _get_new_notifs(self) -> set[str]:
        """Get mm_hashes which got a remote transfer notification."""
        notified_mm_hashes: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                notif_decode = notif.decode("utf-8")
                mm_notif, mm_ratio = notif_decode.rsplit("$", 1)
                mm_hash, tp_ratio = mm_notif.rsplit(":", 1)
                if mm_hash not in self._reqs_to_send:
                    logger.error(
                        "Potentially invalid EC cache for unrecognized "
                        "mm_hash %s was retrieved by a consumer. "
                        "It may have expired.",
                        mm_hash,
                    )
                    continue

                self.mm_consumer_counts_by_req[mm_hash] += 1
                if self.mm_consumer_counts_by_req[mm_hash] == int(tp_ratio) + int(
                    mm_ratio
                ):
                    notified_mm_hashes.add(mm_hash)
                    del self.mm_consumer_counts_by_req[mm_hash]
                    del self._reqs_to_send[mm_hash]
                    logger.debug(f"Removed mm_hash {mm_hash} from send queue after notification")

        return notified_mm_hashes

    def _pop_done_transfers(
        self, transfers: dict[str, list[tuple[int, float]]]
    ) -> set[str]:
        """Pop completed transfers by checking for DONE state."""
        done_mm_hashes: set[str] = set()
        for mm_hash, handles in list(transfers.items()):
            in_progress = False
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    # Transfer completed successfully
                    pass
                elif xfer_state == "PROC":
                    in_progress = True
                    continue
                else:
                    raise RuntimeError(
                        f"EC transfer failed with state {xfer_state}"
                    )
            if not in_progress:
                done_mm_hashes.add(mm_hash)
                del transfers[mm_hash]
        return done_mm_hashes

    def _release_mm_handle(self, mm_hash: str, handle: int):
        """Release NIXL handles and deregister memory for a completed transfer."""
        if (mm_hash, handle) not in self._xfer_side_mm_handle:
            return

        _, src_xfer_handle, dst_xfer_handle = self._xfer_side_mm_handle[
            (mm_hash, handle)
        ]
        _, mm_descs, recv_tensor, _ = self._registered_mm_descs[mm_hash]

        self.nixl_wrapper.release_dlist_handle(src_xfer_handle)
        self.nixl_wrapper.release_dlist_handle(dst_xfer_handle)
        if mm_descs is not None:
            self.nixl_wrapper.deregister_memory(mm_descs)

        # Note: Copy to encoder cache is handled in get_finished via _copy_recv_to_encoder_cache

        del self._xfer_side_mm_handle[(mm_hash, handle)]
        del self._registered_mm_descs[mm_hash]

    def wait_for_load(self) -> None:
        """Wait until all encoder cache transfers are complete."""
        # Poll until all receiving transfers are done
        while self._mm_hashes_need_recv:
            done_recving = self._pop_done_transfers(self._recving_transfers)
            for mm_hash in done_recving:
                if mm_hash in self._recving_metadata:
                    meta = self._recving_metadata.pop(mm_hash)
                    self._copy_recv_to_encoder_cache(mm_hash)
                self._mm_hashes_need_recv.discard(mm_hash)
            
            if self._mm_hashes_need_recv:
                # Small sleep to avoid busy waiting
                time.sleep(0.01)


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket."""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
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
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.encoder_cache_manager import MemorySegment
    from vllm.v1.request import Request

EngineId = str
MMHash = str

GET_META_MSG = b"get_meta_msg"

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
    mm_base_addr: int
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
        mm_base_addr: int,
        remote_host: str,
        remote_port: int,
        remote_engine_id: str,
        tp_size: int,
    ):
        """Add a request to receive encoder cache from remote."""
        self.reqs_to_recv[mm_hash] = ECReqMeta(
            mm_hash=mm_hash,
            num_encoder_tokens=num_encoder_tokens,
            mm_base_addr=mm_base_addr,
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
        ec_cache: torch.Tensor,
    ):
        """Register encoder cache tensors with NIXL."""
        assert self.connector_worker is not None
        # For NIXL, we register the main encoder cache tensor
        # Individual mm_hash caches are handled via recv tensors
        if hasattr(self.connector_worker, "encoder_cache") and \
           self.connector_worker.encoder_cache is not None:
            # Already registered
            return
        # The encoder_cache will be registered when it's first set
        # via register_encoder_cache method
        self.connector_worker.register_encoder_cache(ec_cache)

    def start_load_caches(
        self, encoder_cache, **kwargs
    ) -> None:
        """Start loading encoder caches from remote via NIXL."""
        assert self.connector_worker is not None
        metadata: NixlECConnectorMetadata = self._get_connector_metadata()

        self.connector_worker.start_load_caches(encoder_cache, metadata)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Save encoder cache to remote via NIXL (no-op for NIXL, handled by request_finished)."""
        # NIXL handles saving via request_finished callback
        # This method is called but we don't need to do anything here
        
        
        pass  # hero

        # add mm_base_addr to metadata by assign the data pointer add of the cache
        # TODO: find a better way... this is supposed to be worker side method

        # assert self.connector_scheduler is not None
        # TODO: reg cache per mm
        # TODO: it doesnt work now coz self.connector_scheduler in worker role is None. they are different connector objects
        if mm_hash in encoder_cache: 
            base_addr = encoder_cache[mm_hash].data_ptr()
            self.connector_scheduler._ENCODER_MM_BASE_ADDRS[self.engine_id][mm_hash] = base_addr


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
        self._ENCODER_MM_BASE_ADDRS: dict[EngineId, dict[MMHash, int]]

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
            # Cache exists if remote_mm_segments contains this mm_hash

            # # Hero brute-force, return all true if is consumer
            # has_cache = not self.is_producer
            # logger.debug(f"Hero: has_cache for mm_hash {mm_hash}: {has_cache}")
            # result.append(has_cache)

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
                if mm_hash_params:
                    meta.add_recv_req(
                        mm_hash=mm_hash,
                        num_encoder_tokens=num_encoder_tokens,
                        mm_base_addr=mm_hash_params["mm_base_addr"],
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
        global _ENCODER_MM_BASE_ADDRS   # hero

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

            mm_base_addr = self._ENCODER_MM_BASE_ADDRS.get(self.engine_id, {}).get(mm_hash)
            logger.debug(f"hero: mm_base_addr is {mm_base_addr} for mm_hash {mm_hash}")

            # Return params keyed by mm_hash for proxy aggregation
            # Format: {mm_hash: {do_remote_encode, num_encoder_tokens, remote_engine_id, ...}}
            result_params[mm_hash] = {
                "do_remote_encode": True,
                "num_encoder_tokens": num_encoder_tokens,
                "mm_base_addr": mm_base_addr,
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

        # Encoder cache registration
        self.encoder_cache: Optional[torch.Tensor] = None
        self.enc_base_addr = 0
        self.enc_token_bytes = 0
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

    def register_encoder_cache(self, encoder_cache: torch.Tensor):
        """Register the main encoder cache tensor with NIXL."""
        self.encoder_cache = encoder_cache
        self.enc_base_addr = encoder_cache.data_ptr()
        self.enc_token_bytes = (
            encoder_cache[0].numel() * encoder_cache.element_size()
        )
        enc_size_bytes = encoder_cache.numel() * encoder_cache.element_size()

        caches_data = [(self.enc_base_addr, enc_size_bytes, 0, "")]
        descs = self.nixl_wrapper.get_reg_descs(
            caches_data, self.nixl_memory_type
        )

        logger.debug("Registering encoder cache descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs)
        logger.debug("Done registering encoder cache descs")
        self._registered_descs.append(descs)

        # Start handshake listener for encoder-only instances
        if self.is_producer:
            metadata = NixlECAgentMetadata(
                engine_id=self.engine_id,
                agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                enc_base_addr=self.enc_base_addr,
                enc_token_bytes=self.enc_token_bytes,
            )
            logger.debug(f"hero: metadata at encoder: {metadata}")
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

    @staticmethod
    def _nixl_handshake_listener(
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
                # Decode the message which contains (GET_META_MSG, rank)
                msg, target_tp_rank = msgspec.msgpack.decode(msg)
                logger.debug(
                    "Received message for tp rank %s; msg: %s",
                    target_tp_rank,
                    msg,
                )
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
                logger.debug(f"hero: encoded_data: {encoded_data}")
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
            logger.debug("hero: aaaaaaaaaaaaaaaaaaaaaa")
            metadata_bytes = sock.recv()
            logger.debug("hero: bbbbbbbbbbbbbbbbbbbbbbbb")
            decoder = msgspec.msgpack.Decoder(NixlECAgentMetadata)
            logger.debug("hero: ccccccccccccccccccccccc")
            metadata = decoder.decode(metadata_bytes)
            logger.debug("hero: ddddddddddddddddddddddddddd")
            logger.debug(f"hero: metadata: {metadata}")
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
        logger.debug(f"hero: add_remote_agent end")
        logger.debug(f"hero: added {nixl_agent_meta.enc_base_addr}, {nixl_agent_meta.enc_token_bytes}")
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
            logger.debug(f"hero: after submit")
            self._handshake_futures[remote_engine_id] = fut
            logger.debug(f"hero: after fut")

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    logger.debug(f"hero: del")
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("EC handshake with %s failed", eid)

            logger.debug(f"hero: add_done_callback")
            fut.add_done_callback(done_callback)

        # def request_ready(_f: Future[Any], entry=(mm_hash, meta)):
        #     self._ready_requests.put(entry)
        # hero

        # check handshake success before proceeding with request
        def request_ready(f: Future[Any], entry=(mm_hash, meta)):
            try:
                # check if handshake succeeded
                f.result()
                self._ready_requests.put(entry)
                logger.debug(f"hero: request is ready! entry: {entry}; f.result(): {f.result()}")
                logger.debug(f"hero self._ready_requests.empty() after request ready: {self._ready_requests.empty()}")
            except Exception:
                # handshake failed
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
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()

        self.device_id = max(recv_tensor.get_device(), 0)
        caches_data = [(base_addr, size_bytes, self.device_id, "")]

        descs = self.nixl_wrapper.get_reg_descs(
            caches_data, self.nixl_memory_type
        )
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs)
        logger.debug("Done registering descs")
        self._registered_mm_descs[mm_hash] = (
            base_addr,
            descs,
            recv_tensor,
            local_segments,
        )
        logger.debug(f"hero: self._registered_mm_descs[mm_hash]s for mm_hash {mm_hash} with base_addr {base_addr}")


    def start_load_caches(
        self,
        encoder_cache,
        metadata: NixlECConnectorMetadata,
    ):
        """Start loading encoder caches from remote via NIXL."""

        # Get the metadata
        assert isinstance(metadata, NixlECConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return
        logger.debug(f"hero: start_load_caches: {metadata.reqs_to_recv.items()}")

        # Reference the encoder_cache
        self._encoder_cache_dict = encoder_cache

        # hero
        # # First, register all receive tensors from encoder_cache
        # logger.debug(f"encoder_cache: {encoder_cache}")
        # for mm_hash, meta in metadata.reqs_to_recv.items():
        #     if mm_hash in encoder_cache:
        #         recv_tensor = encoder_cache[mm_hash]
        #         if mm_hash not in self._registered_descs:
        #             logger.debug(
        #                 "Registering receive tensor for mm_hash %s, shape: %s",
        #                 mm_hash,
        #                 recv_tensor.shape,
        #             )
        #             self.register_encoder_recv_tensor(mm_hash, recv_tensor)
        #     else:
        #         logger.warning(
        #             "mm_hash %s not found in encoder_cache, cannot register receive tensor!",
        #             mm_hash,
        #         )

        logger.debug(f"hero: metadata.reqs_to_recv.items(): {metadata.reqs_to_recv.items()}")
        for mm_hash, meta in metadata.reqs_to_recv.items():
            logger.debug(f"hero: {(mm_hash, meta)}")
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
                        logger.debug(f"hero: _background_nixl_handshake after")
                        # time.sleep(10)
                        # logger.debug(f"hero: wait 10 for request ready before read mm and during handshake")
                        continue

            logger.debug(f"hero: before read mm")
            # time.sleep(2)
            # logger.debug(f"hero: wait 2 for request ready before read mm")
            # Handshake completed, start async read transfer
            self._read_mm_segments(mm_hash, meta)

        if metadata.reqs_to_recv:   # if not empty
            for mm_hash, meta in metadata.reqs_to_recv.items():
                remote_engine_id = meta.remote_engine_id
                if remote_engine_id not in self._remote_agents:
                    logger.debug(f"hero: {remote_engine_id} not in {self._remote_agents}")
                    logger.debug(f"hero: start wait 2 for request ready before read mm")
                    time.sleep(2)
                    logger.debug(f"hero: waited 2 for request ready before read mm")

        # time.sleep(2)
        # logger.debug(f"hero: wait 2 for request ready before read mm")
        # Start transfers for requests whose handshakes have finished
        logger.debug(f"hero: self._ready_requests.empty(): {self._ready_requests.empty()}")
        # logger.debug(f"hero: self._ready_requests.get_nowait(): {self._ready_requests.get_nowait()}")
        # logger.debug(f"hero: self._ready_requests.empty(): {self._ready_requests.empty()}")

        while not self._ready_requests.empty():
            logger.debug(f"while not self._ready_requests.empty():")
            self._read_mm_segments(*self._ready_requests.get_nowait())

        # Add to requests waiting to be read
        self._reqs_to_send.update(metadata.reqs_to_send)

    def _read_mm_segments(self, mm_hash: str, meta: ECReqMeta):
        """Read encoder cache from remote via NIXL.
        
        Transfers the entire encoder cache tensor for the given mm_hash.
        """
        logger.debug(f"hero: _read_mm_segments")
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

        if mm_hash not in self._registered_mm_descs:
            if remote_engine_id not in self._remote_enc_base_addr:
                logger.error(
                    "Remote encoder base addr for engine %s not found when "
                    "starting transfer for mm_hash %s",
                    remote_engine_id,
                    mm_hash
                )
                return
            # logger.warning(
            #     "mm_hash %s not registered for receive, skipping transfer",
            #     mm_hash,
            # )
            # return
            # hero
            base_addr, token_bytes = self._remote_enc_base_addr[remote_engine_id]

            # Derive hidden size from bytes-per-token and dtype element size.
            elem_size = torch.tensor([], dtype=self.encoder_cache_dtype).element_size()
            assert token_bytes % elem_size == 0, (
                f"enc_token_bytes {token_bytes} not divisible by element size "
                f"{elem_size} for dtype {self.encoder_cache_dtype}"
            )
            hidden_size = token_bytes // elem_size

            # Allocate local receive tensor and expose it to the encoder_cache dict.
            recv_tensor = torch.empty(
                (num_encoder_tokens, hidden_size),
                device=self.device_type,
                dtype=self.encoder_cache_dtype,
            )

            assert self._encoder_cache_dict is not None
            logger.debug(f"hero: self._encoder_cache_dict: {self._encoder_cache_dict}")
            self._encoder_cache_dict[mm_hash] = recv_tensor

            logger.debug(f"hero: self._encoder_cache_dict after recv_tensor: {self._encoder_cache_dict}")
            
            logger.debug(f"hero: size: {recv_tensor.size(), self._encoder_cache_dict[mm_hash].size()} / recv_tensor for {mm_hash}: {self._encoder_cache_dict[mm_hash]}")

            logger.debug(
                "Allocating receive tensor for mm_hash %s with shape %s "
                "(num_tokens=%d, hidden_size=%d)",
                mm_hash,
                recv_tensor.shape,
                num_encoder_tokens,
                hidden_size,
            )

            self.register_encoder_recv_tensor(mm_hash, recv_tensor)

        base_addr, token_bytes = self._remote_enc_base_addr[remote_engine_id]
        local_base_addr, _, recv_tensor, _ = self._registered_mm_descs[mm_hash]

        tp_ratio = (
            self._tp_size[self.engine_id] // self._tp_size[remote_engine_id]
        )
        notif_id = f"{mm_hash}:{tp_ratio}$1".encode()
        agent_name = self._remote_agents[remote_engine_id][0]

        # Transfer the whole tensor: offset 0, length num_encoder_tokens
        seg_bytes = num_encoder_tokens * token_bytes
        remote_addr = meta.mm_base_addr # base_addr  # Start from base address (offset 0)
        local_addr = local_base_addr  # Start from local base address (offset 0)

        remote_segments = [(remote_addr, seg_bytes, 0)]
        local_seg_addrs = [(local_addr, seg_bytes, 0)]
        idx = [0]  # Single segment

        logger.debug(f"hero: remote_addr: {remote_addr}; local_addr: {local_addr}; num_encoder_tokens: {num_encoder_tokens}; mm_hash: {mm_hash}")

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

        # Handle timeout
        now = time.perf_counter()
        expired_mm_hashes = []
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

        for mm_hash in expired_mm_hashes:
            del self._reqs_to_send[mm_hash]

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
        logger.debug(f"hero: _copy_recv_to_encoder_cache for mm_hash {mm_hash}")
        if mm_hash not in self._registered_mm_descs:
            time.sleep(3) # hero
            if mm_hash not in self._registered_mm_descs:
                logger.debug(f"hero: after 3s mm_hash {mm_hash} still not in {self._registered_mm_descs}")
                return

        _, _, recv_tensor, _ = self._registered_mm_descs[mm_hash]

        # # Copy the whole tensor to encoder_cache[mm_hash]
        # # This matches the shared_storage_connector pattern
        # if self.encoder_cache is not None:
        #     self.encoder_cache[mm_hash] = recv_tensor.clone()
        #     logger.debug(
        #         "Copied received encoder cache for mm_hash %s, shape: %s",
        #         mm_hash,
        #         recv_tensor.shape,
        #     )

        # hero:
        if self._encoder_cache_dict is not None:
            self._encoder_cache_dict[mm_hash] = recv_tensor.clone()
            logger.debug(
                "Copied received encoder cache for mm_hash %s, shape: %s",
                mm_hash,
                recv_tensor.shape,
            )
            logger.debug(f"hero: mm_hash {mm_hash} cloned tensor: {recv_tensor}")

        logger.debug(
            "Encoder cache transfer completed for mm_hash %s, shape: %s",
            mm_hash,
            recv_tensor.shape,
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
                    logger.debug(f"hero: del mm_hash {mm_hash} from self._reqs_to_send")

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
                    self.nixl_wrapper.release_xfer_handle(handle)
                    self._release_mm_handle(mm_hash, handle)
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
        self.nixl_wrapper.deregister_memory(mm_descs)

        # Copy to encoder cache if needed
        if self.encoder_cache is not None:
            self._copy_recv_to_encoder_cache(mm_hash)

        del self._xfer_side_mm_handle[(mm_hash, handle)]
        del self._registered_mm_descs[mm_hash]


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

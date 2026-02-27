# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple buffer manager for Encoder Cache transfer.

This module provides a simple GPU memory buffer manager for storing and
retrieving encoder cache tensors during distributed transfer operations.
"""

import ctypes
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class BufferSlot:
    """Represents an allocated slot in the buffer."""

    offset: int  # Offset from base address in bytes
    size: int  # Size in bytes


class EncoderCacheTransferBuffer:
    """Simple buffer manager for encoder cache transfer.

    This class manages a contiguous memory region for storing encoder
    cache tensors. It uses a simple allocation strategy with LRU eviction
    when the buffer is full.

    Supports both GPU (CUDA) and CPU (pinned) memory for RDMA transfers.

    Args:
        buffer_size: Total size of the buffer in bytes.
        device: Device to allocate the buffer on ("cuda" or "cpu").
            For CPU, pinned memory is used for RDMA compatibility.

    Attributes:
        buffer_size: Total buffer size in bytes.
        base_tensor: The underlying tensor.
        base_address: Base memory address of the buffer.
    """

    def __init__(
        self,
        buffer_size: int,
        device: str = "cuda",
    ):
        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive")

        self.buffer_size = buffer_size
        self.device = device

        # Allocate contiguous memory
        # Use float32 for allocation, actual data can be any dtype
        num_elements = buffer_size // 4  # 4 bytes per float32
        if device == "cpu":
            # Use pinned memory for CPU buffer (required for RDMA)
            self.base_tensor = torch.empty(
                num_elements, dtype=torch.float32, pin_memory=True
            )
        else:
            # GPU memory
            self.base_tensor = torch.empty(
                num_elements, dtype=torch.float32, device=device
            ).contiguous()
        self.base_address = self.base_tensor.data_ptr()

        # Track allocated slots: addr -> BufferSlot
        # Use OrderedDict for LRU eviction (oldest first)
        self._allocated: OrderedDict[int, BufferSlot] = OrderedDict()
        self._lock = threading.Lock()

        # Current allocation offset (simple bump allocator)
        self._next_offset = 0

        # Optional callback when a slot is freed (for bookkeeping sync)
        self.on_free: Callable[[int], None] | None = None

        logger.debug(
            "EncoderCacheTransferBuffer initialized: size=%d bytes, "
            "base_addr=0x%x, device=%s",
            buffer_size,
            self.base_address,
            device,
        )

    def allocate(self, size: int) -> int:
        """Allocate a slot of the given size.

        Args:
            size: Size to allocate in bytes.

        Returns:
            Address of the allocated slot.

        Raises:
            ValueError: If size exceeds buffer capacity.
        """
        if size <= 0:
            raise ValueError("Allocation size must be positive")
        if size > self.buffer_size:
            raise ValueError(
                f"Requested size {size} exceeds buffer capacity {self.buffer_size}"
            )

        with self._lock:
            # Try to allocate at current offset
            if self._next_offset + size <= self.buffer_size:
                addr = self.base_address + self._next_offset
                self._allocated[addr] = BufferSlot(offset=self._next_offset, size=size)
                self._next_offset += size
                logger.debug(
                    "Allocated slot: addr=0x%x, size=%d, next_offset=%d",
                    addr,
                    size,
                    self._next_offset,
                )
                return addr

            # Buffer full, try LRU eviction
            while self._allocated and self._next_offset + size > self.buffer_size:
                self._evict_oldest()

            # Reset if buffer is empty
            if not self._allocated:
                self._next_offset = 0

            # Try again after eviction
            if self._next_offset + size <= self.buffer_size:
                addr = self.base_address + self._next_offset
                self._allocated[addr] = BufferSlot(offset=self._next_offset, size=size)
                self._next_offset += size
                logger.debug(
                    "Allocated slot after eviction: addr=0x%x, size=%d",
                    addr,
                    size,
                )
                return addr

            raise ValueError(f"Cannot allocate {size} bytes, buffer fragmented or full")

    def _evict_oldest(self) -> int:
        """Evict the oldest (LRU) slot.

        Returns:
            Address of the evicted slot.
        """
        if not self._allocated:
            raise ValueError("No slots to evict")

        # Pop oldest entry (first in OrderedDict)
        addr, slot = self._allocated.popitem(last=False)

        logger.debug(
            "Evicted slot: addr=0x%x, size=%d",
            addr,
            slot.size,
        )

        # Invoke callback if set
        if self.on_free is not None:
            self.on_free(addr)

        return addr

    def free(self, addr: int) -> None:
        """Free an allocated slot.

        Args:
            addr: Address of the slot to free.

        Raises:
            ValueError: If address is not allocated.
        """
        with self._lock:
            if addr not in self._allocated:
                raise ValueError(f"Address 0x{addr:x} is not allocated")

            slot = self._allocated.pop(addr)
            logger.debug("Freed slot: addr=0x%x, size=%d", addr, slot.size)

            # Invoke callback if set
            if self.on_free is not None:
                self.on_free(addr)

    def store_tensor(self, tensor: torch.Tensor) -> int:
        """Store a tensor in the buffer.

        Args:
            tensor: Tensor to store. For GPU buffer, must be a CUDA tensor.
                For CPU buffer, can be either CUDA or CPU tensor.

        Returns:
            Address where the tensor is stored.

        Raises:
            ValueError: If tensor device is incompatible or allocation fails.
        """
        # For GPU buffer, source must be CUDA tensor
        if self.device != "cpu" and not tensor.is_cuda:
            raise ValueError(
                f"GPU buffer requires CUDA tensor, got tensor on {tensor.device}"
            )

        size = tensor.numel() * tensor.element_size()
        addr = self.allocate(size)

        # Get the slot info
        with self._lock:
            slot = self._allocated.get(addr)
            if slot is None:
                raise ValueError("Allocation failed")

        # Create a view into the buffer and copy data
        try:
            buffer = (ctypes.c_byte * slot.size).from_address(addr)
            buffer_tensor = torch.frombuffer(
                buffer, dtype=tensor.dtype, count=tensor.numel()
            ).reshape(tensor.shape)
            buffer_tensor.copy_(tensor)
        except Exception as e:
            self.free(addr)
            raise ValueError(f"Failed to store tensor: {e}") from e

        return addr

    def load_tensor(
        self,
        addr: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        copy: bool = True,
    ) -> torch.Tensor:
        """Load a tensor from the buffer.

        Args:
            addr: Address where tensor is stored.
            dtype: Data type of the tensor.
            shape: Shape of the tensor.
            device: Target device for the loaded tensor (required if copy=True).
            copy: If True, copy to device. If False, return view at address.

        Returns:
            The loaded tensor.

        Raises:
            ValueError: If address is invalid or parameters don't match.
        """
        with self._lock:
            if addr not in self._allocated:
                raise ValueError(f"Address 0x{addr:x} is not allocated")
            slot = self._allocated[addr]

        num_elements = 1
        for dim in shape:
            num_elements *= dim

        element_size = torch.tensor([], dtype=dtype).element_size()
        required_size = num_elements * element_size

        if required_size > slot.size:
            raise ValueError(
                f"Requested tensor size {required_size} exceeds slot size {slot.size}"
            )

        # Create view into buffer
        buffer = (ctypes.c_byte * slot.size).from_address(addr)
        buffer_tensor = torch.frombuffer(
            buffer, dtype=dtype, count=num_elements
        ).reshape(shape)

        if not copy:
            return buffer_tensor

        if device is None:
            device = self.device

        target_tensor = torch.empty(shape, dtype=dtype, device=device)
        target_tensor.copy_(buffer_tensor)
        return target_tensor

    def cleanup(self) -> None:
        """Clean up all resources."""
        with self._lock:
            self._allocated.clear()
            self._next_offset = 0
        logger.debug("ECBuffer cleaned up")

    @property
    def used_size(self) -> int:
        """Return the currently used buffer size in bytes."""
        return self._next_offset

    @property
    def free_size(self) -> int:
        """Return the available buffer size in bytes."""
        return self.buffer_size - self._next_offset

    @property
    def num_allocated(self) -> int:
        """Return the number of allocated slots."""
        with self._lock:
            return len(self._allocated)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains a new class `ECStoreBufferBase` that allows developers
to manage the ECCache buffer as a simple key-value storage buffer with basic
put/get operations.

These classes above are abstracted behind class `ECCacheBufferBase`.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

class ECStoreBufferBase(ABC):
    """
    Abstract base class for a ECCache storage buffer with key-value semantics.
    This class provides a simple key-value storage buffer abstract with basic
    put/get operations, which enables flexible ECCache transfer granular
    control.

    The functionality is similar to a distributed key-value store, where:
    - Key: A unique string identifier for the cached entry
    - Value: Tensor to be stored and retrieved
    """
    @abstractmethod
    def close(self) -> None:
        """Close the buffer and release resources.

        This method is responsible for cleaning up resources related to the
        ECCache buffer when it is no longer needed.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def put(
        self,
        key: str,
        value: torch.Tensor,
    ) -> None:
        """Store a key-value pair in the buffer.

        Args:
            key (str): Unique identifier for a tensor, this tensor could be the
                key cache tensor, value cache tensor, or hidden state tensor
                generated during model forwarding.

            value (Optional[torch.Tensor]): Tensor to be stored.
        """
        pass

    @abstractmethod
    def batch_put(
        self,
        keys: List[str],
        values: List[torch.Tensor],
    ) -> None:
        pass

    @abstractmethod
    def get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """Retrieve a value from the buffer by key.

        Args:
            key (str): Unique identifier for a tensor, this tensor could be the
                key cache tensor, value cache tensor, or hidden state tensor
                generated during model forwarding.

        Returns:
            Optional[torch.Tensor]: Stored tensor if exists, None otherwise.
        """
        pass

    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
    ) -> List[Optional[torch.Tensor]]:
        pass

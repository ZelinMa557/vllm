from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, List, Optional, Protocol, Tuple

from vllm.utils import Device

BlockId = int


class CompoundBlock(ABC):

    @abstractmethod
    def append_token_ids(self, token_ids: List[int]) -> None:
        pass

    @property
    @abstractmethod
    def start_block_id(self) -> Optional[int]:
        pass
    
    @property
    @abstractmethod
    def num_sub_blocks(self) -> Optional[int]:
        pass

    @start_block_id.setter
    @abstractmethod
    def start_block_id(self, value: Optional[int]) -> None:
        """NOTE: Do not use this API outside Block."""
        self._start_block_id = value

    @property
    @abstractmethod
    def token_ids(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def num_empty_slots(self) -> int:
        pass

    @property
    @abstractmethod
    def is_full(self) -> bool:
        pass

    @property
    @abstractmethod
    def prev_block(self) -> Optional["CompoundBlock"]:
        pass

    @property
    @abstractmethod
    def computed(self) -> bool:
        raise NotImplementedError

    @computed.setter
    @abstractmethod
    def computed(self, value) -> bool:
        """Should be only used by PrefixCacingAllocator"""
        raise NotImplementedError

    @property
    @abstractmethod
    def last_accessed(self) -> float:
        raise NotImplementedError

    @last_accessed.setter
    @abstractmethod
    def last_accessed(self, last_accessed_ts: float):
        raise NotImplementedError

    class Factory(Protocol):

        @abstractmethod
        def __call__(
            self,
            prev_block: Optional["CompoundBlock"],
            token_ids: List[int],
            sub_block_size: int,
            allocator: "BlockAllocator",
            num_sub_blocks: Optional[int] = None
            start_sub_block_id: Optional[int] = None,
        ) -> "CompoundBlock":
            pass

    @property
    @abstractmethod
    def content_hash(self) -> Optional[int]:
        """Return the content-based hash of the current block, or None if it is
        not yet defined or not supported.

        For the content-based hash to be defined, the current block must be
        full.
        """
        return None


class BlockAllocator(ABC):

    @abstractmethod
    def allocate_mutable(self, prev_block: Optional[CompoundBlock], size: int) -> CompoundBlock:
        pass

    @abstractmethod
    def allocate_immutable(self, prev_block: Optional[CompoundBlock],
                           token_ids: List[int], size: int) -> CompoundBlock:
        pass

    @abstractmethod
    def free(self, block: CompoundBlock) -> None:
        pass

    @abstractmethod
    def fork(self, last_block: CompoundBlock) -> List[CompoundBlock]:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_physical_block_id(self, absolute_id: int) -> int:
        pass

    @abstractmethod
    def swap_out(self, blocks: List[CompoundBlock]) -> None:
        pass

    @abstractmethod
    def swap_in(self, blocks: List[CompoundBlock]) -> None:
        pass

    @property
    @abstractmethod
    def all_block_ids(self) -> FrozenSet[int]:
        pass

    @abstractmethod
    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        pass

    @abstractmethod
    def cow_block_if_not_appendable(self, block: CompoundBlock) -> Optional["BlockId"]:
        """NOTE: This should not be used besides Block"""
        pass

    @abstractmethod
    def promote_to_immutable_block(self, block: CompoundBlock) -> CompoundBlockId:
        """NOTE: This should not be used besides Block"""
        pass

    @abstractmethod
    def get_num_blocks_touched(self,
                               blocks: List[CompoundBlock],
                               num_lookahead_slots: int = 0) -> int:
        pass

    class NoFreeBlocksError(ValueError):
        pass


class DeviceAwareBlockAllocator(ABC):

    @abstractmethod
    def allocate_mutable(self, prev_block: Optional[CompoundBlock],
                         device: Device, size: int) -> CompoundBlock:
        pass

    @abstractmethod
    def allocate_immutable(self, prev_block: Optional[CompoundBlock],
                           token_ids: List[int], device: Device, size: int) -> CompoundBlock:
        pass

    @abstractmethod
    def get_num_free_blocks(self, device: Device) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self, device: Device) -> int:
        pass

    @abstractmethod
    def free(self, block: CompoundBlock) -> None:
        pass

    @abstractmethod
    def fork(self, last_block: CompoundBlock) -> List[CompoundBlock]:
        pass

    @property
    @abstractmethod
    def all_block_ids(self) -> FrozenSet[int]:
        pass

    @abstractmethod
    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        pass

    @abstractmethod
    def get_num_blocks_touched(self,
                               blocks: List[CompoundBlock],
                               device: Device,
                               num_lookahead_slots: int = 0) -> int:
        pass

    @abstractmethod
    def swap(self, blocks: List[CompoundBlock], source_device: Device,
             dest_device: Device) -> Dict[int, int]:
        pass

    @abstractmethod
    def get_physical_block_id(self, device: Device, absolute_id: int) -> int:
        pass

    @abstractmethod
    def allocate_or_get_null_block(self) -> CompoundBlock:
        """
        Null blocks are used as a placeholders for KV cache blocks that have
        been dropped due to sliding window.
        There is at most one null block per allocator.
        """
        pass

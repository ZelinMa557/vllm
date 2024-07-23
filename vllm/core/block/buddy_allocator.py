from typing import FrozenSet, Iterable, List, Optional, Set, Tuple

from vllm.core.block.common import (CopyOnWriteTracker, RefCounter,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import CompoundBlock, BlockAllocator, BlockId, Device, BlockInfo
from vllm.utils import cdiv
from math import log, floor
Refcount = int

MAX_ORDER = 8
class BuddyAllocator(BlockAllocator):
    """A simple block allocator that manages blocks of memory without prefix
    caching.

    Args:
        create_block (Block.Factory): A factory function for creating new
            blocks. This is used when a NaiveBlockAllocator is composed within
            a prefix caching allocator -- the naive block allocator must
            construct prefix caching blocks (but shouldn't know anything else
            about them).
        num_blocks (int): The total number of blocks to manage.
        block_size (int): The size of each block in tokens.
        block_ids (Optional[Iterable[int]], optional): An optional iterable of
            block IDs. If not provided, block IDs will be assigned sequentially
            from 0 to num_blocks - 1.
    """

    def __init__(
        self,
        create_block: CompoundBlock.Factory,
        num_blocks: int,
        block_size: int,
        first_block_id: int
    ):
        self._free_lists : List[List[int]] = [[] for _ in range(MAX_ORDER)]
        remain_blocks = num_blocks
        self._offset = first_block_id
        self._total_blocks = remain_blocks
        self._order_size_mp = [pow(2, i+1) for i in range(MAX_ORDER)]
        self._size_order_mp = {(pow(2, i+1), i) for i in range(MAX_ORDER)}
        while remain_blocks > 0:
            order = min(floor(log(remain_blocks, 2)), MAX_ORDER) - 1
            free_list = self._free_lists[order]
            free_list.append(first_block_id)
            size = self._order_size_mp[order]
            first_block_id += size
            remain_blocks -= size
        self._refcounter = RefCounter()
        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly(),
            allocator=self,
        )
        self._create_block = create_block
        self._block_size = block_size

    def allocate_immutable(self,
                           prev_block: Optional[CompoundBlock],
                           token_ids: List[int],
                           device: Optional[Device] = None) -> "CompoundBlock":
        """Allocates a new immutable block with the given token IDs, linked to
        the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.
            token_ids (List[int]): The token IDs to be stored in the new block.

        Returns:
            Block: The newly allocated immutable block.
        """
        assert device is None
        block = self.allocate_mutable(prev_block=prev_block,\
                                       size = \
                                        (len(token_ids)+self._block_size-1)//self._block_size)
        block.append_token_ids(token_ids)
        return block

    def allocate_mutable(self,
                         prev_block: Optional[CompoundBlock],
                         size: int,
                         device: Optional[Device] = None) -> CompoundBlock:
        """Allocates a new mutable block, linked to the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.

        Returns:
            Block: The newly allocated mutable block.
        """
        assert device is None
        block_id = self._allocate_new_block_id(size)
        return self._create_block(
            prev_block=prev_block,
            token_ids=[],
            sub_block_size=self._block_size,
            allocator=self,
            num_sub_blocks=size,
            start_sub_block_id=block_id
        )

    def free(self, block: CompoundBlock) -> None:
        assert block.start_block_id is not None
        block_id = block.start_block_id
        if self._refcounter.decr(block_id) > 0:
            return
        order: int = self._size_order_mp[block.num_sub_blocks]
        while order < MAX_ORDER:
            free_list = self._free_lists[order]
            if order == MAX_ORDER-1:
                free_list.append(block_id)
                break
            buddy_id = self._buddy_block_id(block_id, order)
            if buddy_id not in free_list:
                free_list.append(block_id)
                break
            index = free_list.index(buddy_id)
            del free_list[index]
            order += 1
            block_id = min(block_id, buddy_id)
        block.start_block_id = None

    def fork(self, last_block: CompoundBlock) -> List[CompoundBlock]:
        """Creates a new sequence of blocks that shares the same underlying
        memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: The new sequence of blocks that shares the same memory
                as the original sequence.
        """
        source_blocks = get_all_blocks_recursively(last_block)

        forked_blocks: List[CompoundBlock] = []
        prev_block = None
        for block in source_blocks:

            # Increment refcount for each block.
            refcount = self._refcounter.incr(block.start_physical_block_id)
            assert refcount != 1, "can't fork free'd block"

            forked_blocks.append(
                self._create_block(
                    prev_block=prev_block,
                    token_ids=block.token_ids,
                    sub_block_size=self._block_size,
                    allocator=self,
                    num_sub_blocks=block.num_sub_blocks,
                    start_sub_block_id=block.start_physical_block_id
                ))
            prev_block = forked_blocks[-1]

        return forked_blocks

    def get_num_free_blocks(self) -> int:
        count = 0
        for (order, free_list) in enumerate(self._free_lists):
            count += self._order_size_mp[order] * len(free_list)
        return count

    def get_num_total_blocks(self) -> int:
        return self._total_blocks

    def _print_internal_states(self):
        for i, free_list in enumerate(self._free_lists):
            print(f"order: {i+1} compound blocks: {len(free_list)} total blocks: {len(free_list)*self._order_size_mp[i]}")
            print("detail:", free_list)

    def _allocate_new_block_id(self, size: int) -> BlockId:
        order: int = self._size_order_mp[size]
        free_list: List[int] = self._free_lists[order]
        if len(free_list) > 0:
            block_id = free_list.pop()
            self._refcounter.incr(block_id)
            return block_id
        
        non_empty_order = order+1
        while len(self._free_lists[non_empty_order]) == 0:
            non_empty_order += 1
            if non_empty_order == MAX_ORDER:
                self._print_internal_states()
                raise BlockAllocator.NoFreeBlocksError()

        while non_empty_order > order:
            split_block_id = self._free_lists[non_empty_order].pop()
            size = self._order_size_mp[non_empty_order]
            non_empty_order -= 1
            free_list = self._free_lists[non_empty_order]
            free_list.append(split_block_id)
            free_list.append(split_block_id + (size // 2))

        block_id = self._free_lists[order].pop()
        self._refcounter.incr(block_id)
        return block_id


    def _buddy_block_id(self, id_, order) -> int:
        id_offset = id_ - self._offset
        higher_order_size = self._order_size_mp[order+1]
        if id_offset % higher_order_size != 0:
            this_order_size = self._order_size_mp[order]
            return id_ - this_order_size
        else:
            return id_


    def get_physical_block_id(self, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain block allocator
        given the absolute block id.

        Args:
            absolute_id (int): The absolute block id for the block 
            in whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return absolute_id - self._offset

    @property
    def refcounter(self):
        return self._refcounter

    def cow_block_if_not_appendable(self, block: CompoundBlock) -> Optional[BlockId]:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            Optional[BlockId]: The block index of the new block if a copy-on
                -write operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        return self._cow_tracker.cow_block_if_not_appendable(block)

    def clear_copy_on_writes(self) -> List[Tuple[BlockId, BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices.
        """
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as computed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        """Determine blocks that can be skipped in prefill.

        Since the naive allocator does not support prefix caching, always return
        an empty list.
        """
        return []

    def promote_to_immutable_block(self, block: CompoundBlock) -> BlockId:
        raise NotImplementedError

    def get_num_blocks_touched(self,
                               blocks: List[CompoundBlock],
                               num_lookahead_slots: int = 0) -> int:
        """Determine the number of blocks that will be touched by
        swapping in/out the given blocks from certain sequence
        group with the provided num_lookahead_slots.

        Args:
            blocks (List[Block]): The potential blocks to swap.
            num_lookahead_slots (int): number of lookahead slots (0 for swap 
                out).
        
        Returns:
            int: the number of blocks that will be touched by
                swapping in/out the given blocks and num_lookahead_slots.
        """
        # NOTE: for naive block, we use set to eliminate common blocks among
        # seqs, also we compare the empty slots in the mutable blocks with
        # lookahead slots to get the number of unique new block that are
        # needed.
        block_id_size_map = {}
        new_block_count = 0
        # TODO(cade): make sure the logic is correct and clean it up.
        for block in blocks:
            if not block.is_full and num_lookahead_slots != 0:
                if block.num_empty_slots < num_lookahead_slots:
                    new_block_count += cdiv(
                        num_lookahead_slots - block.num_empty_slots,
                        self._block_size)
            block_id_size_map[block.start_physical_block_id] = block.num_sub_blocks
        num_touched_blocks = new_block_count + sum(block_id_size_map.values())
        return num_touched_blocks

    def swap_out(self, blocks: List[CompoundBlock]) -> List[BlockInfo]:
        res = [BlockInfo(block.start_physical_block_id, block.num_sub_blocks) for block in blocks]
        for block in blocks:
            self.free(block)
        return res

    def swap_in(self, blocks: List[CompoundBlock]) -> Tuple[List[BlockInfo], CompoundBlock]:
        #TODO(mxk) add compaction here! needs to refact the whole swap logic
        last_alloc = None
        res = []
        for block in blocks:
            if block.is_full:
                alloc = self.allocate_immutable(last_alloc,
                                                block.token_ids, block.num_sub_blocks)
            else:
                alloc = self.allocate_mutable(last_alloc, block.num_sub_blocks)
                alloc.append_token_ids(block.token_ids)
            last_alloc = alloc
            block.start_physical_block_id = alloc.start_physical_block_id
            res.append(BlockInfo(last_alloc.start_physical_block_id, last_alloc.num_sub_blocks))
        return res, last_alloc


class CompoundBlockImpl(CompoundBlock):
    """An implementation of the CompoundBlock class that does not support prefix
    caching.

    The CompoundBlockImpl class represents a block of token IDs with a fixed size. It
    provides methods for appending token IDs to the block and manages copy-on
    -write operations when necessary.

    Args:
        prev_block (Block): The previous block in the sequence.
        token_ids (List[int]): The initial token IDs to be stored in the block.
        block_size (int): The maximum number of token IDs that can be stored in
            the block.
        allocator (BlockAllocator): The block allocator associated with this
            block.
        block_id (Optional[int], optional): The physical block index
            of this block. Defaults to None, which means no allocation has been
            made.
        _cow_target (Optional[Block], optional): The copy-on-write target block.
            If not provided, it defaults to self.
    """

    def __init__(self,
                 prev_block: Optional[CompoundBlock],
                 token_ids: List[int],
                 sub_block_size: int,
                 allocator: BlockAllocator,
                 num_sub_blocks: Optional[int] = None,
                 start_sub_block_id: Optional[int] = None,
                 _cow_target: Optional[CompoundBlock] = None):
        self._token_ids: List[int] = token_ids
        self._sub_block_size = sub_block_size
        self._prev_block = prev_block
        self._allocator = allocator
        self._num_sub_blocks = num_sub_blocks
        self._start_sub_block_id = start_sub_block_id
        self._cow_target = _cow_target if _cow_target is not None else self
        self._num_empty_slots = sub_block_size * num_sub_blocks
        self._num_empty_slots -= len(token_ids)
        assert self._num_empty_slots >= 0
        self._append_token_ids_no_cow(token_ids)

    def append_token_ids(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block, instructing the allocator
        to perform a copy-on-write if necessary.

        Args:
            token_ids (List[int]): The token IDs to be appended to the block.
        """
        self._append_token_ids_no_cow(token_ids)

        if self._start_block_id is not None:
            self._start_block_id = (self._allocator.cow_block_if_not_appendable(
                self._cow_target))

    def _append_token_ids_no_cow(self, token_ids: List[int]) -> None:
        assert self._num_empty_slots >= len(token_ids)
        self._token_ids.extend(token_ids)
        self._num_empty_slots -= len(token_ids)

    @property
    def computed(self) -> bool:
        raise NotImplementedError

    @computed.setter
    def computed(self, value) -> None:
        raise NotImplementedError

    @property
    def last_accessed(self) -> float:
        raise NotImplementedError

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        raise NotImplementedError

    @property
    def is_full(self) -> bool:
        return self._num_empty_slots == 0

    @property
    def num_empty_slots(self) -> int:
        return self._num_empty_slots

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def sub_block_size(self) -> int:
        return self._sub_block_size
    
    @property
    def start_block_id(self) -> Optional[int]:
        return self._start_block_id
    
    @property
    def num_sub_blocks(self) -> Optional[int]:
        return self._num_sub_blocks

    @start_block_id.setter
    def start_block_id(self, value: Optional[int]) -> None:
        """NOTE: Do not use this API outside Block."""
        self._start_block_id = value

    @property
    def prev_block(self) -> Optional["CompoundBlock"]:
        return self._prev_block

    @property
    def content_hash(self) -> Optional[int]:
        return None

    def get_entry(self) -> int:
        return ((self.num_sub_blocks - 1) << 24) + self._start_sub_block_id
    
    @property
    def physical_block_ids(self) -> List[int]:
        return [i for i in range(self._start_sub_block_id, self._start_sub_block_id+\
                                 self.num_sub_blocks)]
    
    @property
    def start_physical_block_id(self) -> Optional[int]:
        return self._start_sub_block_id
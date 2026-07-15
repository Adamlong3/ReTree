import heapq
import os
import time
from functools import lru_cache
from types import SimpleNamespace

from loguru import logger
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import dflash_generate, cuda_time, empty_stage_times


DDTREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")
DDTREE_TREE_BUILD_STAGE_ORDER = (
    "tree_build_copy",
    "tree_build_heap",
    "tree_build_visibility",
)


_CPP_COMPACT_ENABLED = False


@lru_cache(maxsize=1)
def load_cpp_compact_module():
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:
        logger.warning(
            f"torch.utils.cpp_extension is unavailable; falling back to Python cache compaction. {exc}"
        )
        return None

    cpp_source = r"""
torch::Tensor compact_tail_inplace(torch::Tensor cache_tensor, int64_t past_length, torch::Tensor keep_current_indices) {
    TORCH_CHECK(cache_tensor.dim() >= 2, "cache_tensor must have rank >= 2");
    TORCH_CHECK(keep_current_indices.dim() == 1, "keep_current_indices must be a 1D tensor");
    TORCH_CHECK(keep_current_indices.scalar_type() == torch::kLong, "keep_current_indices must have dtype torch.long");
    TORCH_CHECK(cache_tensor.device() == keep_current_indices.device(), "cache_tensor and keep_current_indices must be on the same device");

    const int64_t seq_dim = cache_tensor.dim() - 2;
    TORCH_CHECK(past_length >= 0, "past_length must be non-negative");
    TORCH_CHECK(past_length <= cache_tensor.size(seq_dim), "past_length exceeds cache sequence length");

    const int64_t current_length = cache_tensor.size(seq_dim) - past_length;
    if (current_length <= 0) {
        return cache_tensor;
    }

    const int64_t keep_count = keep_current_indices.numel();
    TORCH_CHECK(keep_count >= 0, "keep_count must be non-negative");
    TORCH_CHECK(keep_count <= current_length, "keep_count exceeds appended window length");

    if (keep_count == 0 || keep_count == current_length) {
        return cache_tensor;
    }

    auto tail = cache_tensor.narrow(seq_dim, past_length, current_length);
    auto kept_tail = tail.index_select(seq_dim, keep_current_indices);
    cache_tensor.narrow(seq_dim, past_length, keep_count).copy_(kept_tail);
    return cache_tensor;
}
"""
    try:
        module = load_inline(
            name="ddtree_compact_tail_ext_v1",
            cpp_sources=[cpp_source],
            functions=["compact_tail_inplace"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("Loaded inline C++ tail cache compaction extension for DDTree.")
        return module
    except Exception as exc:
        logger.warning(
            f"Failed to build inline C++ tail cache compaction extension; falling back to Python implementation. {exc}"
        )
        return None


def maybe_enable_cpp_compact(enabled: bool) -> None:
    global _CPP_COMPACT_ENABLED
    _CPP_COMPACT_ENABLED = enabled
    if enabled:
        load_cpp_compact_module()


def _empty_tree_result(
    build_subtimes: dict[str, float],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
    dict[str, float],
]:
    visibility = torch.zeros((1, 1), dtype=torch.bool)
    visibility[0, 0] = True
    return (
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        [-1],
        [dict()],
        visibility,
        build_subtimes,
    )


def _build_ngram_next_table(
    context_ids: torch.Tensor | list[int] | None,
    max_n: int,
    context_window: int,
) -> dict[tuple[int, ...], dict[int, int]]:
    """
    Build online prefix -> next-token counts from the current visible context.

    This is sample-local/context-local. It does not use ground-truth answers,
    target future tokens, or any offline target trajectory.
    """
    if context_ids is None or max_n < 2:
        return {}

    if isinstance(context_ids, torch.Tensor):
        ctx = context_ids.detach().to(device="cpu", dtype=torch.long).tolist()
    else:
        ctx = list(context_ids)

    if ctx and isinstance(ctx[0], list):
        ctx = ctx[0]

    if context_window > 0 and len(ctx) > context_window:
        ctx = ctx[-context_window:]

    ctx = [int(x) for x in ctx]
    table: dict[tuple[int, ...], dict[int, int]] = {}

    for n in range(2, max_n + 1):
        prefix_len = n - 1
        if len(ctx) <= prefix_len:
            continue
        for i in range(0, len(ctx) - prefix_len):
            prefix = tuple(ctx[i : i + prefix_len])
            nxt = int(ctx[i + prefix_len])
            bucket = table.setdefault(prefix, {})
            bucket[nxt] = bucket.get(nxt, 0) + 1

    return table


def _ngram_bonus_for_token(
    parent_tail: tuple[int, ...],
    token_id: int,
    ngram_table: dict[tuple[int, ...], dict[int, int]],
    max_n: int,
    min_count: int,
    use_log_count: bool,
    longer_weight: float,
    max_bonus: float,
) -> float:
    """Return online n-gram continuity bonus for appending token_id."""
    if not ngram_table or max_n < 2:
        return 0.0

    token_id = int(token_id)
    total = 0.0

    for n in range(2, max_n + 1):
        prefix_len = n - 1
        if len(parent_tail) < prefix_len:
            continue
        prefix = tuple(parent_tail[-prefix_len:])
        count = ngram_table.get(prefix, {}).get(token_id, 0)
        if count < min_count:
            continue

        base = float(np.log1p(count)) if use_log_count else 1.0
        order_weight = float(longer_weight ** max(0, n - 2))
        total += order_weight * base

    if max_bonus > 0:
        total = min(total, max_bonus)
    return float(total)


def _append_tail(
    parent_tail: tuple[int, ...], token_id: int, tail_size: int
) -> tuple[int, ...]:
    if tail_size <= 0:
        return ()
    token_id = int(token_id)
    if len(parent_tail) + 1 <= tail_size:
        return parent_tail + (token_id,)
    return parent_tail[-(tail_size - 1) :] + (token_id,)


def _build_ddtree_tree_original_heap(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
    dict[str, float],
]:
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    sibling_ranks,
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw),
            )

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def _build_ddtree_tree_ngram(
    draft_logits: torch.Tensor,
    budget: int,
    context_ids: torch.Tensor | list[int] | None = None,
    *,
    rank_gated: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
    dict[str, float],
]:
    """
    Online n-gram DDTree.

    Original DDTree score:
        cumulative draft log-prob

    N-gram score:
        cumulative draft log-prob + beta * cumulative online n-gram bonus

    rank_gated=True matches the best stable engineering variant from the
    experiments: only candidate ranks <= DDTREE_NGRAM_RANK_CAP receive n-gram
    bonus. Final tree_budget is unchanged.
    """
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0:
        return _empty_tree_result(build_subtimes)

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    max_n = max(2, int(os.environ.get("DDTREE_NGRAM_MAX_N", "4")))
    beta = float(os.environ.get("DDTREE_NGRAM_BETA", "0.15"))
    rank_cap = int(os.environ.get("DDTREE_NGRAM_RANK_CAP", "8"))
    min_count = int(os.environ.get("DDTREE_NGRAM_MIN_COUNT", "1"))
    context_window = int(os.environ.get("DDTREE_NGRAM_CONTEXT_WINDOW", "2048"))
    use_log_count = os.environ.get("DDTREE_NGRAM_USE_LOG_COUNT", "1") != "0"
    longer_weight = float(os.environ.get("DDTREE_NGRAM_LONGER_WEIGHT", "1.5"))
    max_token_bonus = float(os.environ.get("DDTREE_NGRAM_MAX_TOKEN_BONUS", "6.0"))
    exact_beta0 = os.environ.get("DDTREE_NGRAM_EXACT_BETA0", "1") != "0"

    if abs(beta) < 1e-12 and exact_beta0:
        return _build_ddtree_tree_original_heap(
            draft_logits=draft_logits, budget=budget
        )

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    ngram_table = _build_ngram_next_table(
        context_ids=context_ids,
        max_n=max_n,
        context_window=context_window,
    )

    tail_size = max_n - 1
    if context_ids is None:
        initial_tail: tuple[int, ...] = ()
    else:
        if isinstance(context_ids, torch.Tensor):
            ctx = context_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        else:
            ctx = list(context_ids)
        if ctx and isinstance(ctx[0], list):
            ctx = ctx[0]
        if context_window > 0 and len(ctx) > context_window:
            ctx = ctx[-context_window:]
        initial_tail = tuple(int(x) for x in ctx[-tail_size:])

    bonus_cache: dict[tuple[tuple[int, ...], int], float] = {}

    def raw_bonus(parent_tail: tuple[int, ...], token_id: int) -> float:
        key = (parent_tail, int(token_id))
        cached = bonus_cache.get(key)
        if cached is not None:
            return cached
        value = _ngram_bonus_for_token(
            parent_tail=parent_tail,
            token_id=int(token_id),
            ngram_table=ngram_table,
            max_n=max_n,
            min_count=min_count,
            use_log_count=use_log_count,
            longer_weight=longer_weight,
            max_bonus=max_token_bonus,
        )
        bonus_cache[key] = value
        return value

    def weighted_increment(
        parent_tail: tuple[int, ...], token_id: int, rank: int
    ) -> float:
        if rank_gated and int(rank) > rank_cap:
            return 0.0
        b = raw_bonus(parent_tail, token_id)
        if b <= 0.0:
            return 0.0
        return float(beta * b)

    heap_start = time.perf_counter()

    first_token = int(top_token_ids_np[0, 0])
    first_logw = float(top_log_probs_np[0, 0])
    first_inc = weighted_increment(initial_tail, first_token, rank=0)
    first_aug_score = first_logw + first_inc
    first_tail = _append_tail(initial_tail, first_token, tail_size)

    # Heap state:
    #   (-aug_score, parent_index, depth, rank, draft_logw, bonus_score, parent_tail, node_tail)
    heap: list[
        tuple[float, int, int, int, float, float, tuple[int, ...], tuple[int, ...]]
    ] = [(-first_aug_score, 0, 1, 0, first_logw, first_inc, initial_tail, first_tail)]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        (
            _,
            parent_index,
            depth,
            rank,
            draft_logw,
            bonus_score,
            parent_tail,
            node_tail,
        ) = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sib_rank = rank + 1
            sib_token = int(top_token_ids_np[depth - 1, sib_rank])
            sib_draft_logw = (
                draft_logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, sib_rank])
            )
            cur_inc = weighted_increment(parent_tail, token_id, rank=rank)
            sib_inc = weighted_increment(parent_tail, sib_token, rank=sib_rank)
            sib_bonus_score = bonus_score - cur_inc + sib_inc
            sib_aug_score = sib_draft_logw + sib_bonus_score
            sib_tail = _append_tail(parent_tail, sib_token, tail_size)

            heapq.heappush(
                heap,
                (
                    -sib_aug_score,
                    parent_index,
                    depth,
                    sib_rank,
                    sib_draft_logw,
                    sib_bonus_score,
                    parent_tail,
                    sib_tail,
                ),
            )

        if depth < depth_limit:
            child_depth = depth + 1
            child_rank = 0
            child_token = int(top_token_ids_np[child_depth - 1, child_rank])
            child_draft_logw = draft_logw + float(
                top_log_probs_np[child_depth - 1, child_rank]
            )
            child_inc = weighted_increment(node_tail, child_token, rank=child_rank)
            child_bonus_score = bonus_score + child_inc
            child_aug_score = child_draft_logw + child_bonus_score
            child_tail = _append_tail(node_tail, child_token, tail_size)

            heapq.heappush(
                heap,
                (
                    -child_aug_score,
                    current_index,
                    child_depth,
                    child_rank,
                    child_draft_logw,
                    child_bonus_score,
                    node_tail,
                    child_tail,
                ),
            )

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
    context_ids: torch.Tensor | list[int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
    dict[str, float],
]:
    """
    DDTree tree-construction entry.

    Env:
      DDTREE_TREE_STRATEGY=heap               original DDTree
      DDTREE_TREE_STRATEGY=ngram              full online n-gram, beta=0.15 best in tb64 runs
      DDTREE_TREE_STRATEGY=rank_gated_ngram   n-gram only for ranks <= DDTREE_NGRAM_RANK_CAP

    Recommended current best for CSD integration:
      DDTREE_TREE_STRATEGY=rank_gated_ngram
      DDTREE_NGRAM_BETA=0.15
      DDTREE_NGRAM_RANK_CAP=8
      DDTREE_NGRAM_MAX_N=4
    """
    strategy = os.environ.get("DDTREE_TREE_STRATEGY", "heap").strip().lower()

    if strategy in ("heap", "original", "default"):
        return _build_ddtree_tree_original_heap(
            draft_logits=draft_logits, budget=budget
        )

    if strategy in ("ngram", "ngram_continuity", "continuity"):
        return _build_ddtree_tree_ngram(
            draft_logits=draft_logits,
            budget=budget,
            context_ids=context_ids,
            rank_gated=False,
        )

    if strategy in ("rank_gated_ngram", "fast_rank_gated_ngram"):
        return _build_ddtree_tree_ngram(
            draft_logits=draft_logits,
            budget=budget,
            context_ids=context_ids,
            rank_gated=True,
        )

    raise ValueError(
        f"Unknown DDTREE_TREE_STRATEGY={strategy}. Use one of: heap, ngram, rank_gated_ngram."
    )


def compile_ddtree_tree(
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[
            0,
            0,
            :previous_tree_length,
            previous_tree_start : previous_tree_start + previous_tree_length,
        ] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[
        0, 0, :current_length, past_length : past_length + current_length
    ]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[
        :, :, :current_length, : past_length + current_length
    ]
    return (
        verify_input_ids,
        verify_position_ids,
        attention_mask,
        past_length,
        current_length,
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]], posterior: torch.Tensor
) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def _compact_appended_window(
    cache_tensor: torch.Tensor, past_length: int, keep_current_indices: torch.Tensor
) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count == 0 or keep_count == current_length:
        return

    if _CPP_COMPACT_ENABLED:
        module = load_cpp_compact_module()
        if module is not None:
            module.compact_tail_inplace(cache_tensor, past_length, keep_current_indices)
            return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(
        -2, keep_current_indices
    )
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(
    past_key_values: DynamicCache, past_length: int, keep_current_indices: list[int]
) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(
                keep_current_indices, dtype=torch.long, device=device
            )
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if (
                not hasattr(layer, "keys")
                or layer.keys is None
                or layer.keys.numel() == 0
            ):
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for DDTree cache compaction.")


@torch.inference_mode()
def ddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = (
        None
        if stop_token_ids is None
        else torch.tensor(stop_token_ids, device=model.device)
    )

    verify_input_ids_buffer = torch.empty(
        (1, max_tree_nodes), dtype=torch.long, device=model.device
    )
    verify_position_ids_buffer = torch.empty(
        (1, max_tree_nodes), dtype=torch.long, device=model.device
    )
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty(
        (max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits, temperature
    )
    target_hidden = extract_context_feature(
        output.hidden_states, model.target_layer_ids
    )

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -draft_horizon:, :]
        )
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        tree_context_ids = output_ids[0, : start + 1]
        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
            tree_build_subtimes,
        ) = build_ddtree_tree(
            draft_logits[0], tree_budget, context_ids=tree_context_ids
        )
        stage_times["tree_build"] += cuda_time() - tree_build_start
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
        (
            verify_input_ids,
            verify_position_ids,
            verify_attention_mask,
            previous_tree_start,
            previous_tree_length,
        ) = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(
            accepted_indices, dtype=torch.long, device=verify_input_ids.device
        )
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids
        ).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append(
                {
                    "accepted_indices": [int(index) for index in accepted_indices],
                    "tree": {
                        "node_token_ids": [
                            int(token_id) for token_id in node_token_ids.tolist()
                        ],
                        "node_depths": [int(depth) for depth in node_depths.tolist()],
                        "parents": [int(parent) for parent in parents],
                    },
                }
            )

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
    )

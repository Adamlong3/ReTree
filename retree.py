import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from model.recovery import (
    RecoveryMemory,
    target_logit_consistency_gate,
    token_pair_has_stop,
)
from dflash import dflash_generate, cuda_time, empty_stage_times
from ddtree import (
    build_ddtree_tree,
    compile_ddtree_tree,
    compact_dynamic_cache,
    follow_verified_tree,
    maybe_enable_cpp_compact,
    DDTREE_STAGE_ORDER,
    DDTREE_TREE_BUILD_STAGE_ORDER,
)


def _rank_children_by_target_logits(
    children: dict[int, int],
    logits_2d: torch.Tensor,
    position: int,
) -> list[tuple[int, int]]:
    """
    children: token_id -> tree_node_index
    logits_2d: [tree_len, vocab]
    position: current tree node index

    Return:
        [(child_token_id, child_node_index), ...]
        sorted by target logit descending.
    """
    if len(children) == 0:
        return []

    child_tokens = torch.tensor(
        [int(tok) for tok in children.keys()],
        dtype=torch.long,
        device=logits_2d.device,
    )
    child_scores = logits_2d[position].index_select(0, child_tokens)
    order = torch.argsort(child_scores, descending=True)

    ranked = []
    for j in order.tolist():
        tok = int(child_tokens[j].item())
        ranked.append((tok, children[tok]))
    return ranked


def follow_verified_tree_with_recovery(
    child_maps: list[dict[int, int]],
    posterior: torch.Tensor,
    target_logits: torch.Tensor,
    recovery_memory: RecoveryMemory,
    gate_threshold: float = 0.01,
    online_update: bool = True,
    record_top_k: int = 8,
    rescue_top_k: int = 8,
    disallow_rescue_target_ids: set[int] | None = None,
) -> tuple[list[int], int, int]:
    """
    ReTree verifier with target-gated sibling recovery.

    1. If target posterior token exists in current node's children:
           exact accept.
    2. Else:
           try sibling recovery.
           Candidate child must satisfy:
               recovery_memory[(child_token, target_token)] >= lambda
               and the target-logit consistency gate passes.

    The paper configuration starts from a fixed offline prior and records
    causal online events after snapshotting the current frequency gates.
    """
    posterior_tokens = posterior[0].tolist()
    logits_2d = target_logits[0]

    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])
    num_rescued = 0

    while True:
        children = child_maps[current_index]

        # 1. Standard DDTree exact accept.
        if next_token in children:
            current_index = children[next_token]
            accepted_indices.append(current_index)
            next_token = int(posterior_tokens[current_index])
            continue

        # 2. Exact accept failed and no child can be rescued.
        if len(children) == 0:
            break

        # 2.5 If target wants EOS / stop token, do not rescue.
        # Let target's stop token be committed by the normal DDTree commit path.
        if (
            disallow_rescue_target_ids is not None
            and int(next_token) in disallow_rescue_target_ids
        ):
            break

        # 3. Rank children by target confidence at current parent node.
        ranked_children = _rank_children_by_target_logits(
            children=children,
            logits_2d=logits_2d,
            position=current_index,
        )

        rescue_candidates = ranked_children[:rescue_top_k]

        # 4. Check frequency BEFORE optional online update.
        #    This avoids the current mismatch making itself immediately frequent.
        target_tok = int(next_token)
        candidate_pairs = []

        for child_tok, _ in rescue_candidates:
            child_tok = int(child_tok)
            if not token_pair_has_stop(
                child_tok, target_tok, disallow_rescue_target_ids
            ):
                candidate_pairs.append((child_tok, target_tok))

        frequency_snapshot = recovery_memory.snapshot_frequencies(candidate_pairs)
        frequent_before_update = {
            child_tok: (
                (child_tok, target_tok) in frequency_snapshot
                and frequency_snapshot[(child_tok, target_tok)]
                >= recovery_memory.freq_threshold
            )
            for child_tok, _ in rescue_candidates
        }

        # 5. Optional online recovery-memory update.
        #    Only record top-k non-stop children to avoid polluting memory.
        if online_update:
            recovery_memory.record_online_divergences(
                (child_tok for child_tok, _ in ranked_children),
                target_token=target_tok,
                top_k=record_top_k,
                stop_token_ids=disallow_rescue_target_ids,
            )

        # 6. Target-gated sibling recovery.
        rescued = False

        for child_tok, child_idx in rescue_candidates:
            child_tok = int(child_tok)

            # Stop-token-safe rescue:
            # Do not rescue either direction:
            #   target wants EOS, accept non-EOS
            #   target wants non-EOS, accept EOS
            if token_pair_has_stop(child_tok, target_tok, disallow_rescue_target_ids):
                continue

            if not frequent_before_update.get(child_tok, False):
                continue

            is_safe = target_logit_consistency_gate(
                target_logits=logits_2d,
                draft_token_id=child_tok,
                target_top_token_id=target_tok,
                position=current_index,
                threshold=gate_threshold,
            )

            if not is_safe:
                continue

            current_index = child_idx
            accepted_indices.append(current_index)
            num_rescued += 1
            next_token = int(posterior_tokens[current_index])
            rescued = True
            break

        if not rescued:
            break

    return accepted_indices, next_token, num_rescued


@torch.inference_mode()
def retree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    recovery_memory: RecoveryMemory | None = None,
    gate_threshold: float = 0.01,
    recovery_online_update: bool = True,
    recovery_record_top_k: int = 8,
    recovery_rescue_top_k: int = 8,
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

    disallow_rescue_target_ids = None
    if stop_token_ids is not None:
        disallow_rescue_target_ids = {
            int(tok) for tok in stop_token_ids if tok is not None
        }

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
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0
    total_rescued = 0

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

        if recovery_memory is not None:
            (
                accepted_indices,
                next_token,
                num_rescued,
            ) = follow_verified_tree_with_recovery(
                child_maps=child_maps,
                posterior=posterior,
                target_logits=output.logits,
                recovery_memory=recovery_memory,
                gate_threshold=gate_threshold,
                online_update=recovery_online_update,
                record_top_k=recovery_record_top_k,
                rescue_top_k=recovery_rescue_top_k,
                disallow_rescue_target_ids=disallow_rescue_target_ids,
            )
            total_rescued += num_rescued
        else:
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
        total_rescued=total_rescued,
    )

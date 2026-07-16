import argparse
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import (
    DFlashDraftModel,
    sample,
    extract_context_feature,
    load_and_process_dataset,
)
from model.recovery import RecoveryMemory, token_pair_has_stop
from dflash import cuda_time
from ddtree import (
    build_ddtree_tree,
    compile_ddtree_tree,
    compact_dynamic_cache,
    follow_verified_tree,
    maybe_enable_cpp_compact,
)


def rank_children_by_target_logits(
    children: dict[int, int],
    logits_2d: torch.Tensor,
    position: int,
) -> list[int]:
    """
    Return child token ids sorted by target logit descending.
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

    return [int(child_tokens[j].item()) for j in order.tolist()]


@torch.inference_mode()
def calibrate_retree_recovery(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    recovery_memory: RecoveryMemory,
    tree_budget: int,
    temperature: float = 0.6,
    record_top_k: int = 8,
    stop_token_ids: set[int] | None = None,
) -> int:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)

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

    start = input_ids.shape[1]
    draft_prefill = True
    divergence_count = 0
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

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
        if draft_prefill:
            draft_prefill = False

        tree_context_ids = output_ids[0, : start + 1]
        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
            _,
        ) = build_ddtree_tree(
            draft_logits[0], tree_budget, context_ids=tree_context_ids
        )

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

        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample(output.logits, temperature)
        posterior_tokens = posterior[0].tolist()

        current_index = 0
        next_token = int(posterior_tokens[current_index])

        logits_2d = output.logits[0]
        while True:
            children = child_maps[current_index]

            if len(children) == 0:
                break

            # Do not record divergence pairs whose target token is EOS / stop token.
            # Otherwise recovery memory can learn high-frequency pairs such as
            # "\n\n" -> "<|im_end|>", which may later override the target's
            # stop decision.
            target_tok = int(next_token)

            # Stop-token-safe calibration:
            # If target wants EOS / stop token, do not record child -> stop pairs.
            if stop_token_ids is not None and target_tok in stop_token_ids:
                break

            ranked_child_tokens = rank_children_by_target_logits(
                children=children,
                logits_2d=logits_2d,
                position=current_index,
            )

            # Record only top-k non-stop alternatives to avoid memory pollution.
            recorded = 0
            for child_tok in ranked_child_tokens:
                child_tok = int(child_tok)

                # If exact target child exists, do not record target itself as divergence.
                if child_tok == target_tok:
                    continue

                # Stop-token-safe calibration:
                # Do not record either direction:
                #   X -> EOS
                #   EOS -> X
                if token_pair_has_stop(child_tok, target_tok, stop_token_ids):
                    continue

                recovery_memory.update(child_tok, target_tok)
                divergence_count += 1
                recorded += 1

                if recorded >= record_top_k:
                    break

            # Follow exact path if possible; otherwise stop calibration for this round.
            if next_token in children:
                current_index = children[next_token]
                next_token = int(posterior_tokens[current_index])
            else:
                break

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

        start += len(accepted_indices)

        if start >= max_length:
            break

    return divergence_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-budget", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--record-top-k", type=int, default=8)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    if getattr(draft_config, "fusion_target_layers", None) is None:
        draft_config.fusion_target_layers = [1, 9, 17, 25, 33]

    draft_model = (
        DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            config=draft_config,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if draft_model.mask_token_id is None:
        raise ValueError(
            "draft_model.mask_token_id is None. "
            "Please check draft config dflash_config.mask_token_id."
        )

    vocab_size = target.get_input_embeddings().num_embeddings
    if not (0 <= int(draft_model.mask_token_id) < vocab_size):
        raise ValueError(
            f"Invalid draft_model.mask_token_id={draft_model.mask_token_id}, "
            f"target embedding size={vocab_size}."
        )

    print(
        f"Using draft_model.mask_token_id={draft_model.mask_token_id}, target_vocab_size={vocab_size}"
    )

    dataset = load_and_process_dataset(args.dataset)
    if len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    recovery_memory = RecoveryMemory(freq_threshold=0)

    stop_token_ids = (
        {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else None
    )
    print(f"Calibration stop_token_ids={stop_token_ids}")

    total_divergences = 0
    for idx in tqdm(range(len(dataset)), desc="Calibrating ReTree recovery memory"):
        instance = dataset[idx]
        messages = [{"role": "user", "content": instance["turns"][0]}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        div_count = calibrate_retree_recovery(
            model=draft_model,
            target=target,
            input_ids=input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            recovery_memory=recovery_memory,
            tree_budget=args.tree_budget,
            temperature=args.temperature,
            record_top_k=args.record_top_k,
            stop_token_ids=stop_token_ids,
        )
        total_divergences += div_count

    if args.output_file is None:
        output_file = f"logs/recovery_tb{args.tree_budget}_{args.dataset}_{args.max_samples}samples.json"
    else:
        output_file = args.output_file

    recovery_memory.save(output_file)
    print(f"\n{'='*50}")
    print("ReTree recovery-memory calibration complete!")
    print(f"Tree budget: {args.tree_budget}")
    print(f"Total divergences recorded: {total_divergences}")
    print(f"Recovery-memory unique pairs: {recovery_memory.total_pairs()}")
    print(f"Recovery-memory total rejections: {recovery_memory.total_rejections()}")
    print(f"Recovery memory saved to: {output_file}")
    top10 = recovery_memory.top_k_pairs(10)
    print(f"Top-10 most frequent divergence pairs:")
    for (d_tok, t_tok), freq in top10:
        d_str = tokenizer.decode([d_tok])
        t_str = tokenizer.decode([t_tok])
        print(f"  ({d_tok} -> {t_tok}) freq={freq}  '{d_str}' -> '{t_str}'")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

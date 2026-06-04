import argparse
import random
import re
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from dflash import dflash_generate
from model import DFlashDraftModel, load_and_process_dataset
from model.correction import OnlineCorrectionMemory
from retree import retree_generate
from tree import maybe_enable_cpp_compact


def extract_boxed_answer(text: str) -> str | None:
    m = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if m:
        return m[-1].strip().replace(',', '').replace(' ', '')
    m = re.findall(r'\\boxed\s+(\S+)', text)
    if m:
        return m[-1].strip().replace(',', '').replace(' ', '')
    m = re.findall(r'####\s*(.+)', text)
    if m:
        return m[-1].strip().replace(',', '').replace(' ', '')
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1].replace(',', '')
    return None


def extract_code_answer(text: str) -> str:
    return text.strip()


def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace(',', '').replace(' ', '')
    try:
        return str(float(ans))
    except (ValueError, TypeError):
        return ans.lower()


def compute_accuracy(outputs: list[dict], dataset_name: str) -> float:
    correct = 0
    total = 0
    for item in outputs:
        pred = item.get("pred")
        ref = item.get("ref")
        if pred is None or ref is None:
            continue
        total += 1
        if dataset_name in ("gsm8k", "math500", "aime24", "aime25"):
            if normalize_answer(str(pred)) == normalize_answer(str(ref)):
                correct += 1
        elif dataset_name in ("humaneval", "mbpp"):
            if str(pred).strip() == str(ref).strip():
                correct += 1
        else:
            if str(pred).strip().lower() == str(ref).strip().lower():
                correct += 1
    return correct / total if total > 0 else 0.0


ALL_METHODS = ["dflash", "retree"]
METHOD_LABELS = {
    "dflash": "DFlash",
    "retree": "ReTree",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=int, default=32)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--methods",
        type=str,
        default="dflash,retree",
        help="Comma-separated list of methods to benchmark: dflash,retree",
    )
    parser.add_argument("--correction-freq-threshold", type=int, default=6)
    parser.add_argument("--correction-threshold", type=float, default=0.01)
    parser.add_argument("--correction-memory-file", type=str, default=None)
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--correction-online-update", action="store_true")
    parser.add_argument("--correction-record-top-k", type=int, default=8)
    parser.add_argument("--correction-recover-top-k", type=int, default=8)
    args = parser.parse_args()

    active_methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for method in active_methods:
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {ALL_METHODS}")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    need_tree = "retree" in active_methods
    target_attn_implementation = "sdpa" if need_tree else "flash_attention_2"
    draft_attn_implementation = "flash_attention_2"

    if dist.is_main():
        if need_tree:
            print("ReTree uses a custom tree attention mask on target. Forcing target to sdpa.")
        else:
            print("Target using flash_attention_2 for best DFlash performance.")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    if getattr(draft_config, "fusion_target_layers", None) is None:
        draft_config.fusion_target_layers = [1, 9, 17, 25, 33]

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        config=draft_config,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    correction_memory = None
    if need_tree:
        if args.correction_memory_file is not None:
            correction_memory = OnlineCorrectionMemory.from_file(
                args.correction_memory_file,
                freq_threshold=args.correction_freq_threshold,
            )
            if dist.is_main():
                total_pairs = correction_memory.total_pairs()
                total_rej = correction_memory.total_rejections()
                top5 = correction_memory.top_k_pairs(5)
                print(f"ReTree: loaded correction memory from {args.correction_memory_file}")
                print(f"ReTree: memory pairs={total_pairs}, recorded events={total_rej}")
                print(f"ReTree: memory top-5 pairs={top5}")
                print(f"ReTree: consistency threshold={args.correction_threshold}")
        else:
            correction_memory = OnlineCorrectionMemory(freq_threshold=args.correction_freq_threshold)
            if dist.is_main():
                print(
                    "ReTree: starting with empty correction memory "
                    f"(threshold={args.correction_freq_threshold})"
                )
                print(f"ReTree: consistency threshold={args.correction_threshold}")

    if dist.is_main():
        print(f"Active methods: {[METHOD_LABELS[m] for m in active_methods]}")

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

    if dist.is_main():
        print(f"Using draft_model.mask_token_id={draft_model.mask_token_id}, target_vocab_size={vocab_size}")

    dataset = load_and_process_dataset(args.dataset)

    if args.dataset in ("gsm8k", "math500", "aime24", "aime25"):
        try:
            from datasets import load_dataset as ld

            if args.dataset == "gsm8k":
                raw = ld("openai/gsm8k", "main", split="test")
                answers = []
                for i in range(len(raw)):
                    m = re.findall(r'####\s*(.+)', raw[i]["answer"])
                    answers.append(m[-1].strip().replace(',', '').replace(' ', '') if m else None)
                dataset = dataset.add_column("ref_answer", answers)
            elif args.dataset == "math500":
                raw = ld("HuggingFaceH4/MATH-500", split="test")
                answers = []
                for i in range(len(raw)):
                    ans = str(raw[i]["answer"]).strip().replace(',', '').replace(' ', '') if "answer" in raw[i] else None
                    answers.append(ans)
                dataset = dataset.add_column("ref_answer", answers)
        except Exception:
            pass

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    if "dflash" in active_methods:
        _ = dflash_generate(
            model=draft_model,
            target=target,
            input_ids=warmup_input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=warmup_max_new_tokens,
            block_size=block_size,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
        )
    if need_tree:
        _ = retree_generate(
            model=draft_model,
            target=target,
            input_ids=warmup_input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=warmup_max_new_tokens,
            block_size=block_size,
            tree_budget=args.tree_budget,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
            correction_memory=correction_memory,
            correction_threshold=args.correction_threshold,
            correction_online_update=False,
            correction_record_top_k=args.correction_record_top_k,
            correction_recover_top_k=args.correction_recover_top_k,
        )

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response["baseline"] = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

            if "dflash" in active_methods:
                response["dflash"] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                )

            if need_tree:
                response["retree"] = retree_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    tree_budget=args.tree_budget,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    correction_memory=correction_memory,
                    correction_threshold=args.correction_threshold,
                    correction_online_update=args.correction_online_update,
                    correction_record_top_k=args.correction_record_top_k,
                    correction_recover_top_k=args.correction_recover_top_k,
                )

            for key in active_methods:
                result = response[key]
                gen_ids = result.output_ids[0, result.num_input_tokens:]
                out_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred = None
                if args.dataset in ("gsm8k", "math500", "aime24", "aime25"):
                    pred = extract_boxed_answer(out_text)
                elif args.dataset in ("humaneval", "mbpp"):
                    pred = extract_code_answer(out_text)
                response[f"{key}_pred"] = pred

            response["ref_answer"] = instance.get("ref_answer")

            last_method = active_methods[-1]
            last_output_text = tokenizer.decode(
                response[last_method].output_ids[0, response[last_method].num_input_tokens:],
                skip_special_tokens=True,
            )
            messages.append({"role": "assistant", "content": last_output_text})
            responses.append(response)

    accuracy_records_by_method = {key: [] for key in active_methods}

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    for result in responses:
        for key in active_methods:
            accuracy_records_by_method[key].append({
                "pred": result.get(f"{key}_pred"),
                "ref": result.get("ref_answer"),
            })

    t_baseline = np.mean([result["baseline"].time_per_output_token for result in responses])

    print(f"\n{'=' * 60}")
    for key in active_methods:
        label = METHOD_LABELS[key]
        t_method = np.mean([result[key].time_per_output_token for result in responses])
        speedup = t_baseline / t_method
        avg_accept = np.mean([np.mean(result[key].acceptance_lengths) for result in responses])
        avg_output = np.mean([result[key].num_output_tokens for result in responses])

        acceptance_lengths = list(chain(*[result[key].acceptance_lengths for result in responses]))
        max_al = max(acceptance_lengths) if acceptance_lengths else 0
        histogram = [
            acceptance_lengths.count(b) / len(acceptance_lengths)
            for b in range(min(block_size + 1, max_al + 2))
        ]

        print(f"\n--- {label} ---")
        print(f"Speedup vs baseline: {speedup:.2f}x")
        print(f"Average Acceptance length: {avg_accept:.2f}")
        print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
        print(f"Average Output Length: {avg_output:.2f} tokens/request")

        if args.dataset in ("gsm8k", "math500", "aime24", "aime25"):
            records = accuracy_records_by_method[key]
            acc = compute_accuracy(records, args.dataset)
            valid = sum(1 for rec in records if rec["pred"] is not None and rec["ref"] is not None)
            print(f"Accuracy: {acc * 100:.1f}% ({valid}/{len(records)} evaluable)")

    if need_tree:
        total_recovered = sum(result["retree"].total_recovered for result in responses)
        print(f"ReTree Total Recovered Tokens: {total_recovered}")

    if correction_memory is not None:
        print(f"ReTree Memory Total Pairs: {correction_memory.total_pairs()}")
        print(f"ReTree Memory Total Events: {correction_memory.total_rejections()}")
        print(f"ReTree Memory Top-10 pairs: {correction_memory.top_k_pairs(10)}")

        if args.correction_online_update:
            memory_path = (
                f"logs/retree_memory_{args.dataset}_freq{args.correction_freq_threshold}_"
                f"tau{args.correction_threshold}_online.json"
            )
            correction_memory.save(memory_path)
            print(f"ReTree memory saved to {memory_path}")
        else:
            print("ReTree memory not saved because correction_online_update=False.")

    if dist.is_main():
        print(f"ReTree online update: {args.correction_online_update}")
        print(f"ReTree record_top_k: {args.correction_record_top_k}")
        print(f"ReTree recover_top_k: {args.correction_recover_top_k}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

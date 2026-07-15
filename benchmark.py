import argparse
import random
import re
import time
from itertools import chain
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, sample, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from retree import retree_generate
from model.recovery import RecoveryMemory


def extract_boxed_answer(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m[-1].strip().replace(",", "").replace(" ", "")
    m = re.findall(r"\\boxed\s+(\S+)", text)
    if m:
        return m[-1].strip().replace(",", "").replace(" ", "")
    m = re.findall(r"####\s*(.+)", text)
    if m:
        return m[-1].strip().replace(",", "").replace(" ", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def extract_code_answer(text: str) -> str:
    return text.strip()


def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace(",", "").replace(" ", "")
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


def attach_ref_answer(dataset, dataset_name: str):
    """Attach reference answers from the already-loaded local dataset.

    This avoids reloading Hub datasets such as openai/gsm8k or HuggingFaceH4/MATH-500,
    which breaks in offline/local-cache runs and silently caused 0/N evaluable accuracy.
    """
    if "ref_answer" in dataset.column_names:
        return dataset

    if dataset_name == "gsm8k":
        if "answer" not in dataset.column_names:
            raise ValueError(
                f"gsm8k dataset has no answer column. columns={dataset.column_names}"
            )

        def add_ref(x):
            m = re.findall(r"####\s*(.+)", str(x["answer"]))
            ref = m[-1].strip().replace(",", "").replace(" ", "") if m else None
            return {"ref_answer": ref}

        return dataset.map(add_ref)

    if dataset_name == "math500":
        if "answer" not in dataset.column_names:
            raise ValueError(
                f"math500 dataset has no answer column. columns={dataset.column_names}"
            )

        def add_ref(x):
            return {
                "ref_answer": str(x["answer"]).strip().replace(",", "").replace(" ", "")
            }

        return dataset.map(add_ref)

    if dataset_name in ("aime24", "aime25"):
        if "answer" in dataset.column_names:
            return dataset.map(
                lambda x: {
                    "ref_answer": str(x["answer"])
                    .strip()
                    .replace(",", "")
                    .replace(" ", "")
                }
            )
        if "solution" in dataset.column_names:
            return dataset.map(
                lambda x: {"ref_answer": extract_boxed_answer(str(x["solution"]))}
            )
        raise ValueError(
            f"{dataset_name} has no answer/solution column. columns={dataset.column_names}"
        )

    return dataset


ALL_METHODS = ["dflash", "ddtree", "retree"]
METHOD_LABELS = {
    "dflash": "DFlash (linear SD)",
    "ddtree": "DDTree",
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
        default="dflash,ddtree,retree",
        help="Comma-separated list of methods to benchmark: dflash,ddtree,retree",
    )
    parser.add_argument(
        "--recovery-freq-threshold",
        dest="recovery_freq_threshold",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--recovery-threshold",
        dest="recovery_threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--recovery-memory-file",
        dest="recovery_memory_file",
        type=str,
        default=None,
    )
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument(
        "--recovery-online-update",
        dest="recovery_online_update",
        action="store_true",
    )
    parser.add_argument(
        "--recovery-record-top-k",
        dest="recovery_record_top_k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--recovery-rescue-top-k",
        dest="recovery_rescue_top_k",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    active_methods = [m.strip() for m in args.methods.split(",")]
    active_methods = list(dict.fromkeys(m for m in active_methods if m))
    for m in active_methods:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method '{m}'. Choose from: {ALL_METHODS}")
    if "retree" in active_methods and "ddtree" not in active_methods:
        active_methods.insert(active_methods.index("retree"), "ddtree")

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

    need_tree = any(m in active_methods for m in ("ddtree", "retree"))
    target_attn_implementation = "sdpa"
    draft_attn_implementation = "sdpa"

    if dist.is_main():
        if need_tree:
            print(
                "DDTree uses custom tree attention mask on target. Forcing target to sdpa."
            )
        else:
            print(
                "No DDTree methods. Target using flash_attention_2 for best performance."
            )

    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=target_attn_implementation,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    if getattr(draft_config, "fusion_target_layers", None) is None:
        draft_config.fusion_target_layers = [1, 9, 17, 25, 33]
    if getattr(draft_config, "num_recurrent_steps", None) is None:
        draft_config.num_recurrent_steps = 1

    draft_model = (
        DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            config=draft_config,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    block_size = (
        args.block_size if args.block_size is not None else draft_model.block_size
    )

    need_retree = "retree" in active_methods
    recovery_memory = None
    if need_retree:
        if args.recovery_memory_file is not None:
            recovery_memory = RecoveryMemory.from_file(
                args.recovery_memory_file,
                freq_threshold=args.recovery_freq_threshold,
            )
            if dist.is_main():
                total_pairs = recovery_memory.total_pairs()
                total_rej = recovery_memory.total_rejections()
                top5 = recovery_memory.top_k_pairs(5)
                print(f"Recovery: loaded memory from {args.recovery_memory_file}")
                print(
                    f"Recovery: total pairs={total_pairs}, total rejections={total_rej}"
                )
                print(f"Recovery: top-5 pairs={top5}")
                print(f"Recovery: logit threshold tau={args.recovery_threshold}")
        else:
            recovery_memory = RecoveryMemory(
                freq_threshold=args.recovery_freq_threshold
            )
            if dist.is_main():
                print(
                    "Recovery: starting with empty memory "
                    f"(online mode, lambda={args.recovery_freq_threshold})"
                )
                print(f"Recovery: logit threshold tau={args.recovery_threshold}")

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
        print(
            f"Using draft_model.mask_token_id={draft_model.mask_token_id}, target_vocab_size={vocab_size}"
        )

    dataset = load_and_process_dataset(args.dataset)
    dataset = attach_ref_answer(dataset, args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(
        target.device
    )
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
    if "ddtree" in active_methods:
        _ = ddtree_generate(
            model=draft_model,
            target=target,
            input_ids=warmup_input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=warmup_max_new_tokens,
            block_size=block_size,
            tree_budget=args.tree_budget,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
        )
    if need_retree:
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
            recovery_memory=recovery_memory,
            scg_threshold=args.recovery_threshold,
            recovery_online_update=False,  # never update memory during warmup
            recovery_record_top_k=args.recovery_record_top_k,
            recovery_rescue_top_k=args.recovery_rescue_top_k,
        )

    responses = []
    accuracy_records = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(
                target.device
            )

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

            if "ddtree" in active_methods:
                response["ddtree"] = ddtree_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    tree_budget=args.tree_budget,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                )

            if need_retree:
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
                    recovery_memory=recovery_memory,
                    scg_threshold=args.recovery_threshold,
                    recovery_online_update=args.recovery_online_update,
                    recovery_record_top_k=args.recovery_record_top_k,
                    recovery_rescue_top_k=args.recovery_rescue_top_k,
                )

            for key in active_methods:
                r = response[key]
                gen_ids = r.output_ids[0, r.num_input_tokens :]
                out_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                p = None
                if args.dataset in ("gsm8k", "math500", "aime24", "aime25"):
                    p = extract_boxed_answer(out_text)
                elif args.dataset in ("humaneval", "mbpp"):
                    p = extract_code_answer(out_text)
                response[f"{key}_pred"] = p

            ref = instance.get("ref_answer")

            response["ref_answer"] = instance.get("ref_answer")

            last_method = active_methods[-1]
            last_output_text = tokenizer.decode(
                response[last_method].output_ids[
                    0, response[last_method].num_input_tokens :
                ],
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

    for r in responses:
        for key in active_methods:
            accuracy_records_by_method[key].append(
                {
                    "pred": r.get(f"{key}_pred"),
                    "ref": r.get("ref_answer"),
                }
            )

    t_baseline = np.mean([r["baseline"].time_per_output_token for r in responses])

    print(f"\n{'='*60}")
    for key in active_methods:
        label = METHOD_LABELS[key]
        t_method = np.mean([r[key].time_per_output_token for r in responses])
        speedup = t_baseline / t_method
        avg_accept = np.mean([np.mean(r[key].acceptance_lengths) for r in responses])
        avg_output = np.mean([r[key].num_output_tokens for r in responses])

        acceptance_lengths = list(
            chain(*[r[key].acceptance_lengths for r in responses])
        )
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
            recs = accuracy_records_by_method[key]
            acc = compute_accuracy(recs, args.dataset)
            valid = sum(
                1 for rec in recs if rec["pred"] is not None and rec["ref"] is not None
            )
            print(f"Accuracy: {acc*100:.1f}% ({valid}/{len(recs)} evaluable)")

    total_requests = len(responses)

    if need_retree:
        total_rescued = sum(r["retree"].total_rescued for r in responses)
        print(f"Recovery Total Rescued Tokens: {total_rescued}")

    if recovery_memory is not None:
        print(f"Recovery Memory Total Pairs: {recovery_memory.total_pairs()}")
        print(
            "Recovery Memory Total Rejections Recorded: "
            f"{recovery_memory.total_rejections()}"
        )
        top10 = recovery_memory.top_k_pairs(10)
        print(f"Recovery Memory Top-10 pairs: {top10}")

        if args.recovery_online_update:
            memory_path = (
                f"logs/recovery_{args.dataset}"
                f"_lambda{args.recovery_freq_threshold}"
                f"_tau{args.recovery_threshold}_online.json"
            )
            recovery_memory.save(memory_path)
            print(f"Recovery memory saved to {memory_path}")
        else:
            print("Recovery memory not saved because recovery_online_update=False.")

    if dist.is_main():
        print(f"Recovery online update: {args.recovery_online_update}")
        print(f"Recovery record_top_k: {args.recovery_record_top_k}")
        print(f"Recovery rescue_top_k: {args.recovery_rescue_top_k}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()

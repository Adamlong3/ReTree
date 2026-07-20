import hashlib
import re
from typing import Optional
import unicodedata

import torch
from datasets import (
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)


MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(
            lambda x: {
                "formatted_input": (
                    f"{x['instruction']}\n\nInput:\n{x['input']}"
                    if x["input"]
                    else x["instruction"]
                )
            }
        )
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        )
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = [
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )

    return dataset


def load_calibration_dataset(data_name: str):
    """Load a training-only source used to initialize recovery memory."""
    math_prompt = (
        "{problem}\nPlease reason step by step, and put your final answer "
        "within \\boxed{{}}."
    )

    if data_name == "gsm8k_train":
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        prompt_fmt = (
            "{question}\nPlease reason step by step, and put your final answer "
            "within \\boxed{{}}."
        )
        return dataset.map(
            lambda row: {"turns": [prompt_fmt.format(**row)]},
            remove_columns=dataset.column_names,
        )

    if data_name == "math_train":
        parts = [
            load_dataset("EleutherAI/hendrycks_math", config, split="train")
            for config in MATH_CONFIGS
        ]
        dataset = concatenate_datasets(parts)
        return dataset.map(
            lambda row: {"turns": [math_prompt.format(**row)]},
            remove_columns=dataset.column_names,
        )

    if data_name == "mbpp_train":
        dataset = load_dataset(
            "google-research-datasets/mbpp", "full", split="train"
        )
        return dataset.map(
            lambda row: {"turns": [row["text"]]},
            remove_columns=dataset.column_names,
        )

    if data_name == "cnn_dailymail_train":
        dataset = load_dataset(
            "abisee/cnn_dailymail", "3.0.0", split="train"
        )
        return dataset.map(
            lambda row: {
                "turns": [
                    "Summarize the following news article:\n\n" + row["article"]
                ]
            },
            remove_columns=dataset.column_names,
        )

    raise ValueError(
        f"Unknown calibration dataset {data_name!r}. Choose from "
        "gsm8k_train, math_train, mbpp_train, cnn_dailymail_train."
    )


def normalize_prompt_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def prompt_hashes(turns: list[str] | tuple[str, ...] | str) -> set[str]:
    """Hash individual turns and the ordered multi-turn prompt."""
    if isinstance(turns, str):
        turns = [turns]
    normalized_turns = [normalize_prompt_text(turn) for turn in turns]
    payloads = normalized_turns + ["\n<RETREE_TURN>\n".join(normalized_turns)]
    return {
        hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for payload in payloads
        if payload
    }


def collect_prompt_hashes(dataset) -> set[str]:
    hashes: set[str] = set()
    for row in dataset:
        hashes.update(prompt_hashes(row["turns"]))
    return hashes

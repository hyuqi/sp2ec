#!/usr/bin/env python3
"""Small, fixed-workload submission runner; see README for timing semantics."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import time
import uuid


VIDEO_NUM_FRAMES = 64
VIDEO_MAX_VISUAL_TOKENS = 8192
VISUAL_OUTPUT_TOKENS = 4000
AIME_OUTPUT_TOKENS = 10000
RATIONALE_TARGET_TOKENS = 3800
LLAVA_CONTACT_SHEETS = 4
OPTIMIZE_INTERVAL = 512
MAX_SKIP_FRACTION = 0.6
KNAPSPEC_NUM_ARMS = 10
UCB_DELTA = 0.1
DRAFT_CONFIDENCE_THRESHOLD = 0.7
SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    module: str
    loader: str
    model_type: str
    text_model_type: str
    layers: int


MODELS = {
    "qwen2.5-vl-7b": ModelSpec("Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl", "load_qwen2_5_vl", "qwen2_5_vl", "qwen2_5_vl_text", 28),
    "llava-next-mistral-7b": ModelSpec("llava-hf/llava-v1.6-mistral-7b-hf", "llava_next", "load_llava_next", "llava_next", "mistral", 32),
    "qwen3-vl-4b": ModelSpec("Qwen/Qwen3-VL-4B-Instruct", "qwen3_vl", "load_qwen3_vl", "qwen3_vl", "qwen3_vl_text", 36),
    "llava-1.5-7b": ModelSpec("llava-hf/llava-1.5-7b-hf", "llava", "load_llava", "llava", "llama", 32),
}
DATASETS = {
    "videomme": ("video_mme", "Video-MME"),
    "mmbench": ("mmbench_video", "MMBench-Video"),
    "arkitscenes": ("arkitscenes", "VSI-Bench-ARKitScenes"),
    "aime25": ("aime25", "AIME2025"),
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def positive_float(value):
    value = nonnegative_float(value)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def boolean(value):
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("use true/false or 1/0")


def model_choice(value):
    key = value.lower()
    for alias, spec in MODELS.items():
        if key in {alias, spec.repo_id.lower(), spec.repo_id.rsplit("/", 1)[-1].lower()}:
            return alias
    raise argparse.ArgumentTypeError("supported models: " + ", ".join(MODELS))


def dataset_choice(value):
    key = value.lower().replace("-", "_")
    aliases = {"video_mme": "videomme", "mmbench_video": "mmbench"}
    key = aliases.get(key, key)
    if key not in DATASETS:
        raise argparse.ArgumentTypeError("supported datasets: " + ", ".join(DATASETS))
    return key


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=model_choice, metavar="MODEL", help="alias or full Hugging Face ID; see README")
    parser.add_argument("--dataset", required=True, type=dataset_choice, metavar="DATASET", help="videomme, mmbench, arkitscenes, aime25")
    parser.add_argument("--cossim", action="store_true", help="AR + cosSim/UCBSpec + cosSim/SP2EC")
    parser.add_argument("--knapspec", action="store_true", help="AR + plain KnapSpec + KnapSpec/UCBSpec + KnapSpec/SP2EC")
    parser.add_argument("--gamma", type=positive_int, default=4, help="maximum draft length (default: 4)")
    parser.add_argument("--use-tree", "--use_tree", type=boolean, default=True, metavar="BOOL", help="tree verification (default: true); false selects a chain")
    parser.add_argument("--beta", type=nonnegative_float, default=0.3, help="SP2EC exploration coefficient (default: 0.3)")
    parser.add_argument("--ucb-l", "--ucb_l", type=positive_float, default=10.0, help="UCBSpec L (default: 10)")
    paths = parser.add_argument_group("data, checkpoint, and output locations")
    paths.add_argument("--model-path", default=os.environ.get("MODEL_PATH"), help="local checkpoint; otherwise use --model's HF ID")
    paths.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "datasets")), help="parent dataset directory (default: DATA_ROOT or ./datasets)")
    paths.add_argument("--data-path", type=Path, help="override the selected dataset's directory")
    paths.add_argument("--num-prompts", type=positive_int, default=5, help="number of question prompts (default: 5)")
    paths.add_argument("--start-idx", type=nonnegative_int, default=0, help="start in deterministic, locally matched prompt order")
    paths.add_argument("--output-dir", type=Path, default=Path("bench_out/submission"), help="parent directory; each run gets a unique child")
    paths.add_argument("--check-data", action="store_true", help="load/count prompts and validate the slice without loading a model")
    args = parser.parse_args(argv)
    if not args.cossim and not args.knapspec:
        parser.error("select --cossim, --knapspec, or both")
    return args


def selected_methods(args):
    methods = ["autoregressive"]
    if args.cossim:
        methods.extend(["cossim_ucbspec", "cossim_sp2ec"])
    if args.knapspec:
        methods.extend(["knapspec", "knapspec_ucbspec", "knapspec_sp2ec"])
    return methods


def output_budget(dataset):
    return AIME_OUTPUT_TOKENS if dataset == "aime25" else VISUAL_OUTPUT_TOKENS


def load_examples(args):
    from data import get_dataset

    dataset_format, directory = DATASETS[args.dataset]
    data_path = args.data_path or args.data_root / directory
    if args.data_path is None and not data_path.exists():
        alternate = {"arkitscenes": "ARKitScenes", "aime25": "AIME25"}.get(args.dataset)
        if alternate and (args.data_root / alternate).exists():
            data_path = args.data_root / alternate
    # Only AIME may use Hub data if its implicit local directory is absent.
    if not data_path.exists():
        if args.dataset == "aime25" and args.data_path is None:
            data_path = None
        else:
            raise FileNotFoundError(f"Dataset directory not found: {data_path}. Set --data-path or --data-root.")
    examples = get_dataset(
        dataset_format=dataset_format,
        num_samples=None,
        random_shuffle=False,
        seed=SEED,
        data_path=str(data_path) if data_path else None,
        videomme_with_rationale=True,
        videomme_rationale_target_tokens=RATIONALE_TARGET_TOKENS,
        videomme_local_only=True,
        mmbench_video_rationale_target_tokens=RATIONALE_TARGET_TOKENS,
        mmbench_video_local_only=True,
        arkitscenes_rationale_target_tokens=RATIONALE_TARGET_TOKENS,
        arkitscenes_local_only=True,
    )
    end = args.start_idx + args.num_prompts
    print(f"[DATA] Available prompts: {len(examples)}; requested [{args.start_idx}:{end}]")
    if end > len(examples):
        raise ValueError(f"Need {end} available prompts, found {len(examples)}. Check local media or lower --num-prompts.")
    return examples[args.start_idx:end], str(data_path) if data_path else "Hugging Face: opencompass/AIME2025"


def generator_kwargs(method, args, coefficients):
    if method == "autoregressive":
        return {}
    kwargs = dict(gamma=args.gamma, tree=args.use_tree, optimize_interval=OPTIMIZE_INTERVAL,
                  draft_confidence_threshold=DRAFT_CONFIDENCE_THRESHOLD,
                  dynamic_draft_stopping=True, include_zero_skip_arm=True)
    if method.startswith("cossim_"):
        kwargs["max_skip_fraction"] = MAX_SKIP_FRACTION
    else:
        kwargs.update(coefficients=coefficients, dp_budget_fraction=MAX_SKIP_FRACTION,
                      num_arms=KNAPSPEC_NUM_ARMS, scoring_mode="legacy")
    if method == "knapspec":
        # Original non-bandit path: directly use the legacy predicted-best DP
        # mask, without constructing/exploring a bandit pool or adding a zero arm.
        kwargs.update(enable_sp2ec=False, num_arms=1, include_zero_skip_arm=False)
    if method.endswith("ucbspec"):
        kwargs.update(ucb_reward_bound=args.ucb_l, ucb_delta=UCB_DELTA)
    elif method.endswith("sp2ec"):
        kwargs["beta"] = args.beta
    return kwargs


def load_environment(args):
    import torch
    from transformers import AutoConfig
    from device_utils import model_input_device, select_device_map
    from utils import Env

    if not torch.cuda.is_available():
        raise RuntimeError("This throughput experiment requires a CUDA GPU.")
    spec = MODELS[args.model]
    checkpoint = args.model_path or spec.repo_id
    config = AutoConfig.from_pretrained(checkpoint)
    text_config = getattr(config, "text_config", config)
    if config.model_type != spec.model_type or text_config.num_hidden_layers != spec.layers:
        raise ValueError(f"{checkpoint} does not match the selected supported model {spec.repo_id}.")
    # In older configs Qwen's nested model_type may be absent; LLaVA's must
    # identify the requested Mistral/Llama backbone, not a different checkpoint.
    if spec.module.startswith("llava") and text_config.model_type != spec.text_model_type:
        raise ValueError(f"Expected a {spec.text_model_type} text backbone, found {text_config.model_type}.")
    dtype = "bfloat16" if all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count())) else "float16"
    loader = getattr(importlib.import_module(spec.module), spec.loader)
    kwargs = dict(dtype=dtype, device_map=select_device_map("cuda"), attn_implementation="sdpa")
    if spec.module == "llava_next":
        kwargs["tree_verification"] = args.use_tree
    model, processor = loader(checkpoint, **kwargs)
    model.eval()
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    env = Env(model=model, tok=tok, device=str(model_input_device(model)),
              eos_id=tok.eos_token_id, pad_id=tok.pad_token_id, processor=processor)
    return env, dtype


def generation_kwargs(args, example, reference_ids):
    metadata = example.metadata or {}
    visual = args.dataset != "aime25"
    llava_visual = visual and args.model.startswith("llava")
    return dict(prompt=example.input, max_new_tokens=output_budget(args.dataset),
                min_new_tokens=output_budget(args.dataset), sample=False,
                reference_output_ids=reference_ids,
                image=metadata.get("image_path"), video=metadata.get("video_path"),
                video_num_frames=VIDEO_NUM_FRAMES,
                video_max_visual_tokens=VIDEO_MAX_VISUAL_TOKENS,
                llava_video_contact_sheets=LLAVA_CONTACT_SHEETS,
                repetition_penalty=1.1 if llava_visual else 1.0,
                no_repeat_ngram_size=8 if llava_visual else 0)


def answer_metrics(dataset, example, text):
    from data import (extract_aime_answer, extract_arkitscenes_answer,
                      extract_mmbench_video_answer, extract_multiple_choice_answer,
                      vsi_mean_relative_accuracy)
    reference = str(example.output).strip()
    correct = mra = None
    if dataset == "aime25":
        predicted = extract_aime_answer(text)
        correct = predicted == reference
    elif dataset == "videomme":
        predicted = extract_multiple_choice_answer(text)
        correct = predicted == reference.upper()
    elif dataset == "mmbench":
        predicted = extract_mmbench_video_answer(text)
        # No claim to implement the official semantic-judge evaluation.
    else:
        answer_type = (example.metadata or {}).get("answer_type", "numerical")
        predicted = extract_arkitscenes_answer(text, answer_type)
        if answer_type == "multiple_choice":
            correct = predicted == reference.upper()
        else:
            mra = vsi_mean_relative_accuracy(predicted, reference)
    return dict(reference_answer=reference, predicted_answer=predicted, is_correct=correct, mra_score=mra)


def empirical_best_arms(history):
    """Best observed pooled TPS within each rebuilt arm set, not an oracle."""
    best = []
    for epoch, snapshot in enumerate(history or []):
        observed = [arm for arm in snapshot.get("arms", [])
                    if arm.get("pulls", 0) > 0 and arm.get("empirical_throughput") is not None]
        if observed:
            winner = max(observed, key=lambda arm: arm["empirical_throughput"])
            best.append(dict(epoch=epoch, arm_index=winner["index"], budget=winner.get("budget"),
                             empirical_tps=winner["empirical_throughput"],
                             pulls=winner["pulls"], skip_set=winner.get("skip_set")))
    return best


def summarize(samples):
    rates = [row["tokens_per_sec"] for row in samples]
    tokens = sum(row["num_output_tokens"] for row in samples)
    measured = sum(row["total_time"] for row in samples)
    elapsed = sum(row["end_to_end_latency_sec"] for row in samples)
    result = dict(num_prompts=len(samples),
                  avg_tokens_per_sec=statistics.mean(rates) if rates else None,
                  tokens_per_sec_population_variance=statistics.pvariance(rates) if rates else None,
                  avg_tokens_per_sec_micro=tokens / measured if measured else None,
                  end_to_end_tokens_per_sec_micro=tokens / elapsed if elapsed else None,
                  avg_output_tokens=tokens / len(samples) if samples else None)
    for key in ("acceptance_rate", "draft_time", "verify_time", "optimization_time", "accepted_length", "is_correct", "mra_score"):
        values = [row[key] for row in samples if row.get(key) is not None]
        result["avg_" + key] = statistics.mean(values) if values else None
    return result


def save_json(path, payload):
    """Replace only this run's own checkpoint, so interrupted runs keep results."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    examples, data_path = load_examples(args)
    if args.check_data:
        print(f"[OK] {args.num_prompts} prompts available for {args.dataset}; no model loaded.")
        return
    if args.dataset != "aime25":
        for command in ("ffmpeg", "ffprobe"):
            if not shutil.which(command):
                raise RuntimeError(f"{command} must be installed and on PATH for visual datasets.")
    # Same bounded FFmpeg reader as the working experiments, not full-video decode.
    os.environ["KNAPSPEC_VIDEO_READER"] = "ffmpeg"
    import numpy as np
    import torch
    from profile_modules import profile_model
    from self_speculation_strategy.autoregressive_generator import AutoregressiveGenerator
    from self_speculation_strategy.knapspec_generator import KnapspecGenerator
    from self_speculation_strategy.sp2ec_basic_generator import SP2ECBasicGenerator
    from self_speculation_strategy.sp2ec_knapspec_generator import SP2ECKnapspecGenerator
    from self_speculation_strategy.vanilla_ucb_generator import VanillaUCBBaseGenerator, VanillaUCBKnapspecGenerator

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env, dtype = load_environment(args)
    context = getattr(getattr(env.model.config, "text_config", env.model.config), "max_position_embeddings", None)
    coefficients = profile_model(env.model) if args.knapspec else None
    methods = selected_methods(args)
    method_settings = {method: generator_kwargs(method, args, coefficients) for method in methods}
    classes = dict(autoregressive=AutoregressiveGenerator, cossim_ucbspec=VanillaUCBBaseGenerator,
                   cossim_sp2ec=SP2ECBasicGenerator, knapspec=KnapspecGenerator,
                   knapspec_ucbspec=VanillaUCBKnapspecGenerator,
                   knapspec_sp2ec=SP2ECKnapspecGenerator)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = args.output_dir / f"{args.model}_{args.dataset}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = dict(run_id=run_id, status="running", model=asdict(MODELS[args.model]),
                    checkpoint=args.model_path or MODELS[args.model].repo_id, dataset=args.dataset,
                    data_path=data_path, num_prompts=args.num_prompts, start_idx=args.start_idx,
                    methods=methods, method_settings=method_settings,
                    gamma=args.gamma, use_tree=args.use_tree, beta=args.beta, ucb_l=args.ucb_l,
                    video_num_frames=VIDEO_NUM_FRAMES, video_max_visual_tokens=VIDEO_MAX_VISUAL_TOKENS,
                    requested_output_tokens=output_budget(args.dataset), min_new_tokens=output_budget(args.dataset),
                    llava_contact_sheets=LLAVA_CONTACT_SHEETS, max_skip_fraction=MAX_SKIP_FRACTION,
                    knapspec_dp_budget_fraction=MAX_SKIP_FRACTION, knapspec_num_arms=KNAPSPEC_NUM_ARMS,
                    include_zero_skip_arm=True, optimize_interval=OPTIMIZE_INTERVAL,
                    knapspec_scoring="legacy", legacy_scoring_draft_lengths=list(range(18)),
                    dynamic_draft_stopping=True, draft_confidence_threshold=DRAFT_CONFIDENCE_THRESHOLD,
                    ucb_delta=UCB_DELTA, seed=SEED, dtype=dtype, context_window=context,
                    profiling_coefficients=coefficients, verification_mode="reference_assisted",
                    timing="model generation including prefill and online arm construction, excluding media preprocessing and one-time profiling",
                    end_to_end_timing="one generate() call including media preprocessing; excludes loading and profiling",
                    gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    hf_device_map=getattr(env.model, "hf_device_map", None), package_versions=versions)
    save_json(output_dir / "manifest.json", manifest)
    print(f"[RUN] {methods}\n[OUT] {output_dir}\n[MODE] Reference-assisted throughput comparison; legacy KnapSpec scoring.")
    records = {method: [] for method in methods}

    def synchronize():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    try:
        with torch.inference_mode():
            # One untimed short AR warm-up avoids charging CUDA/model first-use
            # initialization only to the first measured AR prompt.
            warmup = generation_kwargs(args, examples[0], None)
            warmup.update(max_new_tokens=8, min_new_tokens=8)
            AutoregressiveGenerator(env).generate(**warmup)
            synchronize()
            for index, example in enumerate(examples, start=args.start_idx):
                reference_ids = None
                for method in methods:
                    generator = classes[method](env=env, **method_settings[method])
                    synchronize()
                    start = time.perf_counter()
                    result = generator.generate(**generation_kwargs(args, example, reference_ids))
                    synchronize()
                    elapsed = time.perf_counter() - start
                    record = asdict(result)
                    if method == "autoregressive":
                        reference_ids = list(result.output_ids or [])
                    matches = result.output_ids == reference_ids
                    requested = output_budget(args.dataset)
                    effective = min(requested, max(0, int(context) - result.num_input_tokens)) if context else requested
                    record.update(prompt_index=index, sample_id=str((example.metadata or {}).get("id", index)),
                                  prompt=example.input, metadata=example.metadata,
                                  requested_output_tokens=requested, effective_output_limit=effective,
                                  context_limited=effective < requested,
                                  matches_reference_workload=matches, end_to_end_latency_sec=elapsed,
                                  tokens_per_sec=result.num_output_tokens / result.total_time if result.total_time else 0.0,
                                  accepted_length=result.total_accepted_length / result.total_steps if result.total_steps else None,
                                  empirical_best_arms=empirical_best_arms(result.arm_set_history))
                    record.update(answer_metrics(args.dataset, example, result.text))
                    records[method].append(record)
                    summary = summarize(records[method])
                    baseline = summarize(records["autoregressive"])
                    if summary["avg_tokens_per_sec_micro"] and baseline["avg_tokens_per_sec_micro"]:
                        summary["speedup_vs_ar_micro"] = summary["avg_tokens_per_sec_micro"] / baseline["avg_tokens_per_sec_micro"]
                    save_json(output_dir / f"{method}.json", dict(method=method, manifest="manifest.json", generator_settings=method_settings[method], summary=summary, samples=records[method]))
                    print(f"\n[{method}] prompt {index}: {result.num_output_tokens}/{requested} tokens, {record['tokens_per_sec']:.2f} TPS, reference_match={matches}")
                    if not result.num_output_tokens:
                        raise RuntimeError("No output generated: check the prompt's context length; saved the result for inspection.")
                    if not matches:
                        raise RuntimeError(f"{method} did not reproduce the AR reference workload; saved results, stopping invalid comparison.")
                    del generator
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output_dir / "manifest.json", manifest)
    print(f"\nDone. Results: {output_dir}")


if __name__ == "__main__":
    main()

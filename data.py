# data.py
import json
import random
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
import pandas as pd

# for language modeling problems how long to use the prefix as
PREFIX_LENGTH: int = 100
SPECBENCH_DEFAULT_PATH = "data/spec_bench/question.jsonl"
GOVREPORT_DEFAULT_PATH = "data/govreport/govreport_16K.jsonl"
PG19_DEFAULT_PATH = "data/pg19/pg19_16K.jsonl"
BOOKSUM_DEFAULT_PATH = "data/booksum/booksum_16K.jsonl"

@dataclass
class EvaluationExample:
    input: str
    output: str
    metadata: Optional[Dict] = None


class DatasetFormat:
    AIME24: str = "aime24"
    AIME25: str = "aime25"
    GOVREPORT: str = "govreport"
    PG19: str = "pg19"
    BOOKSUM: str = "booksum"
    GPQA: str = "gpqa"
    MMLU_PRO: str = "mmlu_pro"
    VIDEO_MME: str = "video_mme"
    MMBENCH_VIDEO: str = "mmbench_video"
    ARKITSCENES: str = "arkitscenes"


AIME24_DEFAULT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only the answer) inside \\boxed{{}}.\n\n{message}\n"
    " <think> Let's think step by step. </think>\n"
)
AIME25_DEFAULT_TEMPLATE = (
    "Solve the following AIME problem. Give a complete step-by-step derivation, checking all cases and "
    "calculations carefully. Do not state the final answer early. End with a separate final line containing "
    "the answer, and only the answer, inside \\boxed{{}}.\n\n{message}\n\nSolution:\n"
)


def get_valid_dataset_formats():
    """Get all available dataset formats."""
    return [value for key, value in DatasetFormat.__dict__.items() if not key.startswith('__')]


def apply_template(message: str, template: str = None) -> str:
    """
    Applies a template to a given message.
    
    Parameters:
        message (str): The message to insert into the template.
        template (str): The template with a placeholder for the message in `{message}`.
        
    Returns:
        str: The formatted message with the template applied.
    """
    if template is None:
        return message
    return template.format(message=message)


def normalize_aime_answer(value: str) -> str:
    """Normalize an AIME annotation to its integer answer."""
    match = re.search(r"[-+]?\d+", str(value))
    return str(int(match.group(0))) if match else str(value).strip()


def extract_aime_answer(text: str) -> str:
    """Extract the last boxed or explicitly labeled integer answer."""
    boxed = re.findall(r"\\boxed\s*\{\s*([-+]?\d+)\s*\}", text)
    if boxed:
        return normalize_aime_answer(boxed[-1])
    explicit = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*([-+]?\d+)\s*$", text)
    return normalize_aime_answer(explicit[-1]) if explicit else ""


def prepare_aime24_format(template: str = None) -> List[EvaluationExample]:
    """
    Prepare AIME 2024 dataset (HuggingFaceH4/aime_2024).
    """
    evaluation_data_points = []
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    if template is None:
        template = AIME24_DEFAULT_TEMPLATE

    for i, dp in enumerate(ds):
        problem = dp["problem"]
        answer = str(int(dp["answer"]))
        prompt = apply_template(message=problem, template=template)
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=answer,
                metadata={"id": i, "type": "math_reasoning_aime24"}
            )
        )
    return evaluation_data_points


def prepare_aime25_format(
    template: str = None,
    data_path: Optional[str] = None,
) -> List[EvaluationExample]:
    """
    Prepare AIME 2025 dataset ("opencompass/AIME2025").
    Loads both AIME I and AIME II.
    """
    evaluation_data_points = []
    rows = []

    if data_path:
        root = Path(data_path).expanduser()
        if root.is_file():
            annotation_files = [root]
        elif root.is_dir():
            official_files = [root / "aime2025-I.jsonl", root / "aime2025-II.jsonl"]
            annotation_files = [path for path in official_files if path.is_file()]
            if not annotation_files:
                annotation_files = sorted(root.glob("*.jsonl"))
        else:
            raise FileNotFoundError(f"AIME25 data path does not exist: {root}")
        if not annotation_files:
            raise FileNotFoundError(f"No AIME25 JSONL files were found under: {root}")

        for annotation_path in annotation_files:
            competition = "AIME2025-II" if "ii" in annotation_path.stem.lower() else "AIME2025-I"
            with annotation_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append((json.loads(line), competition))
    else:
        try:
            ds_1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
            ds_2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        except ValueError:
            ds_1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="train")
            ds_2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="train")
        rows.extend((row, "AIME2025-I") for row in ds_1)
        rows.extend((row, "AIME2025-II") for row in ds_2)

    if template is None:
        template = AIME25_DEFAULT_TEMPLATE

    for i, (dp, competition) in enumerate(rows):
        problem = dp["question"]
        raw_answer = dp.get("answer", "")
        answer = normalize_aime_answer(raw_answer)

        prompt = apply_template(message=problem, template=template)
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=answer,
                metadata={
                    "id": i,
                    "type": "math_reasoning_aime25",
                    "competition": competition,
                }
            )
        )
    return evaluation_data_points


def prepare_govreport_format(
    data_path: str = GOVREPORT_DEFAULT_PATH,
    template: str = None,
) -> List[EvaluationExample]:
    evaluation_data_points = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            prompt_text = row.get("text", "")
            # Add instruction for summarization
            prompt_text = f"Summarize the following text:\n\n{prompt_text}\n\nSummary:"
            answer_text = row.get("answer", "")

            prompt = apply_template(message=prompt_text, template=template)

            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=answer_text if str(answer_text).startswith(" ") else f" {answer_text}",
                    metadata={
                        "id": row.get("id", i),
                        "type": "govreport",
                        "original_length": row.get("original_length", None),
                        "trunc_length": row.get("trunc_length", None),
                    },
                )
            )
    return evaluation_data_points


def prepare_pg19_format(
    data_path: str = PG19_DEFAULT_PATH,
    template: str = None,
) -> List[EvaluationExample]:
    evaluation_data_points = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            prompt_text = row.get("text", "")
            # Add instruction for summarization
            prompt_text = f"Summarize the following text:\n\n{prompt_text}\n\nSummary:"
            answer_text = row.get("answer", "")

            prompt = apply_template(message=prompt_text, template=template)

            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=answer_text if str(answer_text).startswith(" ") else f" {answer_text}",
                    metadata={
                        "id": row.get("id", i),
                        "type": "pg19",
                        "original_length": row.get("original_length", None),
                        "trunc_length": row.get("trunc_length", None),
                    },
                )
            )
    return evaluation_data_points


def prepare_booksum_format(
    data_path: str = BOOKSUM_DEFAULT_PATH,
    template: str = None,
) -> List[EvaluationExample]:
    evaluation_data_points = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            prompt_text = row.get("text", "")
            # Add instruction for summarization
            prompt_text = f"Summarize the following text:\n\n{prompt_text}\n\nSummary:"
            answer_text = row.get("answer", "")

            prompt = apply_template(message=prompt_text, template=template)

            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=answer_text if str(answer_text).startswith(" ") else f" {answer_text}",
                    metadata={
                        "id": row.get("id", i),
                        "type": "booksum",
                        "original_length": row.get("original_length", None),
                        "trunc_length": row.get("trunc_length", None),
                    },
                )
            )
    return evaluation_data_points


def prepare_gpqa_format(template: str = None) -> List[EvaluationExample]:
    """Prepare GPQA dataset."""
    evaluation_data_points = []
    # Using 'train' split as per user reference
    try:
        dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond', split='train')
    except Exception as e:
        print(f"Warning: Could not load GPQA dataset: {e}")
        return []

    for i, row in enumerate(dataset):
        choices = [
            row['Correct Answer'],
            row['Incorrect Answer 1'], 
            row['Incorrect Answer 2'], 
            row['Incorrect Answer 3'], 
        ]
        indexed_choices = list(enumerate(choices))
        random.shuffle(indexed_choices)

        labels = ['A', 'B', 'C', 'D']
        shuffled_text = ''
        correct_label = ''
        for j, (original_idx, text) in enumerate(indexed_choices):
            shuffled_text += f"{labels[j]}) {text}\n"
            if original_idx == 0:
                correct_label = labels[j]

        prompt_content = f"""Answer the following multiple-choice question. Think step-by-step before providing your final answer.
    
        Question: {row['Question']}

        Options:
        {shuffled_text}
        The last line of your response MUST be in the following format: "Answer: [LETTER]" where [LETTER] is one of {{A, B, C, D}}."""
        
        prompt = apply_template(message=prompt_content, template=template)
        
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=correct_label,
                metadata={"id": i, "type": "gpqa"}
            )
        )
    return evaluation_data_points


def prepare_mmlu_pro_format(template: str = None) -> List[EvaluationExample]:
    """
    Prepare MMLU-Pro dataset ("TIGER-Lab/MMLU-Pro").
    Ref: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    """
    evaluation_data_points = []
    try:
        # Load validation set (smaller than test set)
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    except Exception as e:
        print(f"Warning: Could not load MMLU-Pro dataset: {e}")
        return []

    for i, row in enumerate(dataset):
        # Fields: question, options, answer, answer_index, cot_content, category
        question = row["question"]
        options = row["options"]
        raw_answer = row["answer"]  # e.g. "C"
        
        # Format options as A, B, C...
        option_text = ""
        labels = []
        for idx, opt in enumerate(options):
            label = chr(65 + idx) # A, B, C...
            labels.append(label)
            option_text += f"{label}. {opt}\n"
        
        prompt_content = (
            f"Question:\n{question}\n\n"
            f"Options:\n{option_text}\n"
            "Answer the question by selecting the correct option. "
            "Think step by step and then provide your final answer in the format: \"Answer: [OPTION]\"."
        )
        
        prompt = apply_template(message=prompt_content, template=template)
        
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=raw_answer,
                metadata={
                    "id": row.get("question_id", i),
                    "type": "mmlu_pro",
                    "category": row.get("category", "unknown") 
                }
            )
        )

    return evaluation_data_points


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def extract_multiple_choice_answer(text: str) -> str:
    """Extract the last explicit A-D answer from a Video-MME response."""
    explicit = re.findall(r"(?i)answer\s*[:\-]?\s*\(?([A-D])\)?", text)
    if explicit:
        return explicit[-1].upper()
    standalone = re.findall(r"(?i)(?<![A-Z])([A-D])(?![A-Z])", text.strip())
    return standalone[-1].upper() if standalone else ""


def _index_video_files(data_path: str) -> Dict[str, Path]:
    root = Path(data_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Video media directory does not exist: {root}")

    index: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            index.setdefault(path.stem, path.resolve())
    if not index:
        raise FileNotFoundError(f"No supported video files were found under: {root}")
    return index


def extract_mmbench_video_answer(text: str) -> str:
    """Extract the answer line requested by the long MMBench-Video prompt."""
    short_answer = re.search(r"(?im)^\s*short answer\s*:\s*(.+?)\s*$", text)
    if short_answer:
        return short_answer.group(1).strip()
    final_answers = re.findall(r"(?im)^\s*final answer\s*:\s*(.+?)\s*$", text)
    if final_answers:
        return final_answers[-1].strip()
    return ""


def extract_arkitscenes_answer(text: str, answer_type: str) -> str:
    """Extract the leading answer requested by the ARKitScenes/VSI-Bench prompt."""
    if answer_type == "multiple_choice":
        explicit = re.search(r"(?im)^\s*answer\s*:\s*\(?([A-D])\)?\s*$", text)
        if explicit:
            return explicit.group(1).upper()
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return first_line.upper() if re.fullmatch(r"(?i)\(?[A-D]\)?", first_line) else ""

    explicit = re.search(
        r"(?im)^\s*answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)\b",
        text,
    )
    if explicit:
        return explicit.group(1).replace(",", "")
    return ""


def vsi_mean_relative_accuracy(prediction: str, ground_truth: str) -> float:
    """Compute VSI-Bench MRA over confidence thresholds 0.50 through 0.95."""
    number_pattern = r"[-+]?\d[\d,]*(?:\.\d+)?"
    predicted_match = re.search(number_pattern, str(prediction))
    ground_truth_match = re.search(number_pattern, str(ground_truth))
    if predicted_match is None or ground_truth_match is None:
        return 0.0

    predicted_value = float(predicted_match.group(0).replace(",", ""))
    ground_truth_value = float(ground_truth_match.group(0).replace(",", ""))
    if ground_truth_value == 0.0:
        return 0.0

    relative_error = abs(predicted_value - ground_truth_value) / abs(ground_truth_value)
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    return sum(relative_error < (1.0 - threshold) for threshold in thresholds) / len(thresholds)


def prepare_video_mme_format(
    data_path: str,
    template: str = None,
    with_rationale: bool = False,
    rationale_target_tokens: Optional[int] = None,
    local_only: bool = False,
    max_videos: Optional[int] = None,
) -> List[EvaluationExample]:
    """Prepare Video-MME questions and resolve their locally extracted videos."""
    if not data_path:
        raise ValueError("data_path is required for Video-MME")
    if rationale_target_tokens is not None:
        if rationale_target_tokens < 1:
            raise ValueError("rationale_target_tokens must be positive")
        if not with_rationale:
            raise ValueError("rationale_target_tokens requires with_rationale=True")

    video_index = _index_video_files(data_path)
    dataset = load_dataset("lmms-lab/Video-MME", "videomme", split="test")
    evaluation_data_points = []
    selected_video_ids = set()

    for row_idx, row in enumerate(dataset):
        video_id = str(row["videoID"])
        video_path = video_index.get(video_id)
        if local_only and video_path is None:
            continue
        if max_videos is not None and video_id not in selected_video_ids:
            if len(selected_video_ids) >= max_videos:
                continue
            selected_video_ids.add(video_id)

        options = "\n".join(str(option) for option in row["options"])
        if rationale_target_tokens is not None:
            instruction = (
                "Select the best answer to the following multiple-choice question based on the video. "
                "Begin with exactly 'Answer: X', where X is A, B, C, or D, so your choice is recorded "
                "before the long analysis. Then write a detailed video evidence report targeting at least "
                f"{rationale_target_tokens} generated tokens. Use each heading below exactly once and in order:\n"
                "1. Question interpretation\n"
                "2. Video and sampling limitations\n"
                "3. Chronological panel-by-panel inventory\n"
                "4. People and identities\n"
                "5. Objects, locations, and visible text\n"
                "6. Actions and scene changes\n"
                "7. Temporal and causal relationships\n"
                "8. Evidence for option A\n"
                "9. Evidence for option B\n"
                "10. Evidence for option C\n"
                "11. Evidence for option D\n"
                "12. Uncertainty and contradictory evidence\n"
                "13. Final synthesis\n"
                "Add only new observations or reasoning in each section. Do not repeat a sentence, pad the "
                "response with symbols, or invent visual evidence. When the frames do not establish a fact, "
                "state that limitation and distinguish visual evidence from outside knowledge. End by repeating "
                "exactly 'Answer: X' with the same choice."
            )
        elif with_rationale:
            instruction = (
                "Select the best answer to the following multiple-choice question based on the video. "
                "Explain your reasoning, then end your response with exactly 'Answer: X', where X is A, B, C, or D."
            )
        else:
            instruction = (
                "Select the best answer to the following multiple-choice question based on the video. "
                "Respond with only the letter (A, B, C, or D) of the correct option."
            )
        prompt_content = (
            f"{instruction}\n\n"
            f"{row['question']}\n"
            f"{options}\n\n"
            f"{'The best answer is:' if not with_rationale else 'Response:'}"
        )
        prompt = apply_template(message=prompt_content, template=template)
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=str(row["answer"]).strip().upper(),
                metadata={
                    "id": row["question_id"],
                    "type": "video_mme",
                    "video_path": str(video_path) if video_path is not None else None,
                    "video_id": row["video_id"],
                    "videoID": video_id,
                    "duration": row["duration"],
                    "domain": row["domain"],
                    "sub_category": row["sub_category"],
                    "task_type": row["task_type"],
                    "url": row["url"],
                    "with_rationale": with_rationale,
                    "rationale_target_tokens": rationale_target_tokens,
                    "row_idx": row_idx,
                },
            )
        )
    return evaluation_data_points


def prepare_mmbench_video_format(
    data_path: str,
    template: str = None,
    rationale_target_tokens: Optional[int] = None,
    local_only: bool = False,
) -> List[EvaluationExample]:
    """Prepare the official MMBench-Video TSV and resolve locally restored videos."""
    if not data_path:
        raise ValueError("data_path is required for MMBench-Video")
    if rationale_target_tokens is not None and rationale_target_tokens < 1:
        raise ValueError("rationale_target_tokens must be positive")

    root = Path(data_path).expanduser()
    annotation_candidates = (
        [root] if root.is_file() else [root / "MMBench-Video.tsv", root.parent / "MMBench-Video.tsv"]
    )
    annotation_path = next((path for path in annotation_candidates if path.is_file()), None)
    if annotation_path is None:
        raise FileNotFoundError(
            f"MMBench-Video.tsv was not found at or above {root}. "
            "Pass the dataset root or the TSV path with --data-path."
        )

    media_root = annotation_path.parent if root.is_file() else root
    video_index = _index_video_files(str(media_root))
    evaluation_data_points = []

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t", quotechar='"')
        for row_idx, row in enumerate(rows):
            video_name = str(row.get("video") or "").strip()
            annotated_path = str(row.get("video_path") or "").strip()
            video_stem = Path(annotated_path).stem or video_name
            video_path = video_index.get(video_stem) or video_index.get(video_name)
            if local_only and video_path is None:
                continue

            question = str(row.get("question") or "").strip()
            if rationale_target_tokens is not None:
                instruction = (
                    "Answer the free-form question using only evidence from the video. Begin with one line in "
                    "exactly this format: 'Short answer: <your concise answer>'. Then write a detailed evidence "
                    f"analysis targeting at least {rationale_target_tokens} generated tokens. Describe relevant "
                    "events chronologically, identify people, objects, actions, text, scene changes, and temporal "
                    "or causal relationships, and explain how the evidence supports the answer. Continue until "
                    "the requested analysis length is reached; do not end after a short explanation. Finish with "
                    "one line in exactly this format: 'Final answer: <the same concise answer>'."
                )
            else:
                instruction = (
                    "Answer the free-form question using only evidence from the video. Begin with one line in "
                    "exactly this format: 'Short answer: <your concise answer>'. Explain the relevant video "
                    "evidence, then finish with 'Final answer: <the same concise answer>'."
                )

            prompt_content = f"{instruction}\n\nQuestion: {question}\n\nResponse:"
            prompt = apply_template(message=prompt_content, template=template)
            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=str(row.get("answer") or "").strip(),
                    metadata={
                        "id": str(row.get("index") or row_idx),
                        "type": "mmbench_video",
                        "video_path": str(video_path) if video_path is not None else None,
                        "video": video_name,
                        "videoID": video_stem,
                        "video_type": str(row.get("video_type") or "").strip(),
                        "dimensions": str(row.get("dimensions") or "").strip(),
                        "rationale_target_tokens": rationale_target_tokens,
                        "row_idx": row_idx,
                    },
                )
            )
    return evaluation_data_points


def prepare_arkitscenes_format(
    data_path: str,
    template: str = None,
    rationale_target_tokens: Optional[int] = None,
    local_only: bool = False,
) -> List[EvaluationExample]:
    """Prepare the ARKitScenes-sourced rows from the official VSI-Bench JSONL."""
    if not data_path:
        raise ValueError("data_path is required for ARKitScenes")
    if rationale_target_tokens is not None and rationale_target_tokens < 1:
        raise ValueError("rationale_target_tokens must be positive")

    root = Path(data_path).expanduser()
    annotation_candidates = [root] if root.is_file() else [root / "test.jsonl", root.parent / "test.jsonl"]
    annotation_path = next((path for path in annotation_candidates if path.is_file()), None)
    if annotation_path is None:
        raise FileNotFoundError(
            f"VSI-Bench test.jsonl was not found at or above {root}. "
            "Pass the extracted ARKitScenes VSI-Bench root or JSONL path with --data-path."
        )

    media_root = annotation_path.parent if root.is_file() else root
    video_index = _index_video_files(str(media_root))
    evaluation_data_points = []

    with annotation_path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("dataset") or "").lower() != "arkitscenes":
                continue

            scene_name = str(row.get("scene_name") or "").strip()
            video_path = video_index.get(scene_name)
            if local_only and video_path is None:
                continue

            raw_options = row.get("options")
            if isinstance(raw_options, dict):
                options = [str(value) for value in raw_options.values()]
            elif isinstance(raw_options, list):
                options = [str(value) for value in raw_options]
            else:
                options = []
            answer_type = "multiple_choice" if options else "numerical"

            if answer_type == "multiple_choice":
                answer_format = "a single option letter"
                answer_example = "X"
                options_text = "\n\nOptions:\n" + "\n".join(options)
            else:
                answer_format = "a single numerical value, including the requested unit when applicable"
                answer_example = "<number>"
                options_text = ""

            if rationale_target_tokens is not None:
                minimum_section_words = max(
                    40,
                    (3 * rationale_target_tokens + 39) // 40,
                )
                instruction = (
                    f"Answer this spatial video question with {answer_format}. Begin with one line in exactly "
                    f"this format: 'Answer: {answer_example}'. Then write a detailed spatial evidence analysis "
                    f"targeting at least {rationale_target_tokens} generated tokens. After the opening answer, write "
                    "all ten numbered sections below in order:\n"
                    "1. Task interpretation and coordinate frame\n"
                    "2. Global scene inventory\n"
                    "3. Camera path and viewpoint changes\n"
                    "4. Target object identification and localization\n"
                    "5. Spatial relationships to stable landmarks\n"
                    "6. Direction, depth, distance, size, or counting evidence\n"
                    "7. Temporal consistency across sampled moments\n"
                    "8. Alternative answers or estimates and counterevidence\n"
                    "9. Occlusion, ambiguity, and error checks\n"
                    "10. Independent reconstruction and final synthesis\n"
                    f"Write at least {minimum_section_words} words in every section. Do not merge, abbreviate, or "
                    "omit sections, even when one is less directly applicable; use it for an independent cross-check. "
                    "Separate direct visual observations from inferences, revisit multiple moments in the video, and "
                    "do not use the phrase 'Final answer' until all ten sections are complete. Finish with one "
                    f"line in exactly this format: 'Final answer: {answer_example}', using the same answer."
                )
            else:
                instruction = (
                    f"Answer this spatial video question with {answer_format}. Begin with 'Answer: {answer_example}', "
                    f"briefly explain the video evidence, and finish with 'Final answer: {answer_example}'."
                )

            question = str(row.get("question") or "").strip()
            prompt_content = f"{instruction}\n\nQuestion: {question}{options_text}\n\nResponse:"
            prompt = apply_template(message=prompt_content, template=template)
            sample_id = row.get("id") if row.get("id") is not None else row.get("idx", row_idx)
            ground_truth = row.get("ground_truth")
            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=str(ground_truth if ground_truth is not None else "").strip(),
                    metadata={
                        "id": str(sample_id),
                        "type": "arkitscenes_vsi_bench",
                        "video_path": str(video_path) if video_path is not None else None,
                        "videoID": scene_name,
                        "scene_name": scene_name,
                        "question_type": str(row.get("question_type") or "").strip(),
                        "answer_type": answer_type,
                        "options": options,
                        "pruned": bool(row.get("pruned", False)),
                        "rationale_target_tokens": rationale_target_tokens,
                        "row_idx": row_idx,
                    },
                )
            )
    return evaluation_data_points


def get_dataset(
    dataset_format: str,
    num_samples: int = None,
    random_shuffle: bool = True,
    seed: int = 42,
    data_path: Optional[str] = None,
    n_shot: int = 0,
    prompt_field: str = "prompt",
    response_field: str = "response",
    template: str = None,
    videomme_with_rationale: bool = False,
    videomme_rationale_target_tokens: Optional[int] = None,
    videomme_local_only: bool = False,
    videomme_max_videos: Optional[int] = None,
    mmbench_video_rationale_target_tokens: Optional[int] = None,
    mmbench_video_local_only: bool = False,
    arkitscenes_rationale_target_tokens: Optional[int] = None,
    arkitscenes_local_only: bool = False,
) -> List[EvaluationExample]:
    """
    Get dataset based on format specification.
    
    Args:
        dataset_format: One of the DatasetFormat values
        num_samples: Number of samples to return (None for all)
        random_shuffle: Whether to shuffle the dataset
        seed: Random seed for reproducibility
        data_path: Path to custom dataset file (for CUSTOM_JSONL)
        n_shot: Number of few-shot examples (for applicable datasets)
        prompt_field: Field name for prompts in custom dataset
        response_field: Field name for responses in custom dataset
        template: Template to apply to prompts
        
    Returns:
        List of EvaluationExample objects
    """
    random.seed(seed)
    
    if dataset_format == DatasetFormat.AIME24:
        evaluation_data_points = prepare_aime24_format(template=template)
    elif dataset_format == DatasetFormat.AIME25:
        evaluation_data_points = prepare_aime25_format(template=template, data_path=data_path)
    elif dataset_format == DatasetFormat.VIDEO_MME:
        evaluation_data_points = prepare_video_mme_format(
            data_path=data_path,
            template=template,
            with_rationale=videomme_with_rationale,
            rationale_target_tokens=videomme_rationale_target_tokens,
            local_only=videomme_local_only,
            max_videos=videomme_max_videos,
        )
    elif dataset_format == DatasetFormat.MMBENCH_VIDEO:
        evaluation_data_points = prepare_mmbench_video_format(
            data_path=data_path,
            template=template,
            rationale_target_tokens=mmbench_video_rationale_target_tokens,
            local_only=mmbench_video_local_only,
        )
    elif dataset_format == DatasetFormat.ARKITSCENES:
        evaluation_data_points = prepare_arkitscenes_format(
            data_path=data_path,
            template=template,
            rationale_target_tokens=arkitscenes_rationale_target_tokens,
            local_only=arkitscenes_local_only,
        )
    elif dataset_format == DatasetFormat.GOVREPORT:
        evaluation_data_points = prepare_govreport_format(
            data_path=data_path if data_path is not None else GOVREPORT_DEFAULT_PATH,
            template=template,
        )
    elif dataset_format == DatasetFormat.PG19:
        evaluation_data_points = prepare_pg19_format(
            data_path=data_path if data_path is not None else PG19_DEFAULT_PATH,
            template=template,
        )
    elif dataset_format == DatasetFormat.BOOKSUM:
        evaluation_data_points = prepare_booksum_format(
            data_path=data_path if data_path is not None else BOOKSUM_DEFAULT_PATH,
            template=template,
        )
    elif dataset_format == DatasetFormat.GPQA:
        evaluation_data_points = prepare_gpqa_format(template=template)
    elif dataset_format == DatasetFormat.MMLU_PRO:
        evaluation_data_points = prepare_mmlu_pro_format(template=template)
    else:
        raise NotImplementedError(f"Unknown dataset format: {dataset_format}")

    if random_shuffle:
        random.shuffle(evaluation_data_points)

    if num_samples is not None and num_samples > 0:
        evaluation_data_points = evaluation_data_points[:num_samples]

    return evaluation_data_points


def get_dataset_info(dataset_format: str) -> Dict:
    """Get information about a dataset format."""
    info = {
        DatasetFormat.AIME24: {
            "description": "AIME 2024 math reasoning (integer answers in \\boxed{})",
            "task_type": "math_reasoning",
            "supports_few_shot": False,
            "typical_input_length": "50-300 tokens",
            "typical_output_length": "1-20 tokens"
        },
        DatasetFormat.AIME25: {
            "description": "AIME 2025 math reasoning (integer answers in \\boxed{})",
            "task_type": "math_reasoning",
            "supports_few_shot": False,
            "typical_input_length": "50-300 tokens",
            "typical_output_length": "1-20 tokens"
        },
        DatasetFormat.GOVREPORT: {
            "description": "GovReport long-document summarization (local JSONL)",
            "task_type": "summarization",
            "supports_few_shot": False,
            "typical_input_length": "long (up to ~16K context)",
            "typical_output_length": "variable"
        },
        DatasetFormat.PG19: {
            "description": "PG19 long-context dataset (local JSONL)",
            "task_type": "long_context",
            "supports_few_shot": False,
            "typical_input_length": "long (up to ~16K context)",
            "typical_output_length": "variable"
        },
        DatasetFormat.BOOKSUM: {
            "description": "BookSum long-context summarization (local JSONL)",
            "task_type": "summarization",
            "supports_few_shot": False,
            "typical_input_length": "long (up to ~16K context)",
            "typical_output_length": "variable"
        },
        DatasetFormat.GPQA: {
            "description": "GPQA difficult multiple-choice QA",
            "task_type": "multiple_choice",
            "supports_few_shot": False,
            "typical_input_length": "100-300 tokens",
            "typical_output_length": "multiple choice answer"
        },
        DatasetFormat.MMLU_PRO: {
            "description": "MMLU-Pro advanced reasoning/knowledge dataset",
            "task_type": "multiple_choice",
            "supports_few_shot": False,
            "typical_input_length": "variable",
            "typical_output_length": "multiple choice answer"
        },
        DatasetFormat.VIDEO_MME: {
            "description": "Video-MME multiple-choice video understanding",
            "task_type": "video_multiple_choice",
            "supports_few_shot": False,
            "typical_input_length": "video plus 50-300 text tokens",
            "typical_output_length": "1 token official, variable with rationale"
        },
        DatasetFormat.MMBENCH_VIDEO: {
            "description": "MMBench-Video long-form free-response video understanding",
            "task_type": "video_free_form_qa",
            "supports_few_shot": False,
            "typical_input_length": "video plus 20-100 text tokens",
            "typical_output_length": "short official answer, variable with evidence analysis"
        },
        DatasetFormat.ARKITSCENES: {
            "description": "ARKitScenes-sourced spatial video QA from VSI-Bench",
            "task_type": "video_spatial_qa",
            "supports_few_shot": False,
            "typical_input_length": "egocentric video plus spatial question",
            "typical_output_length": "one official answer, variable with evidence analysis"
        }
    }

    return info.get(dataset_format, {"description": "Unknown dataset format"})


# Backward compatibility functions
def build_prompt(example: dict, dataset_name: str) -> str:
    """Legacy function for backward compatibility."""
    if dataset_name == "cnn_dm":
        article = example.get("article") or example.get("text") or ""
        return f"Summarize the following article.\n\nArticle:\n{article}\n\nSummary:"
    if dataset_name == "gsm8k":
        q = example.get("question") or example.get("prompt") or ""
        return f"Q: {q}\nA:"
    return example.get("text") or example.get("content") or str(example)


def load_dataset_convenience(dataset: str, num_samples: int):
    """Legacy function for backward compatibility."""
    if dataset == "cnn_dm":
        return get_dataset(DatasetFormat.CNN_DM_SUMMARIZATION, num_samples=num_samples)
    elif dataset == "gsm8k":
        return get_dataset(DatasetFormat.GSM8K, num_samples=num_samples)
    else:
        raise ValueError("Use get_dataset() function with DatasetFormat for new code")

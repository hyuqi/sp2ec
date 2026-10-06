# SP²EC : Adaptive Self-Speculative Decoding for Vision-Language Models

Official implementation of the paper SP²EC : Adaptive Self-Speculative Decoding for Vision-Language Models.

This repository implements cosSim and KnapSpec layer skipping with UCBSpec and
SP2EC selection for vision-language models.

## Setup

Use Linux, Python 3.10, a CUDA-capable NVIDIA GPU.

```bash
conda create -n sp2ec-sample python=3.10 -y
conda activate sp2ec-sample
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```


## Supported models

| `--model` | Hugging Face checkpoint |
| --- | --- |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `llava-next-mistral-7b` | `llava-hf/llava-v1.6-mistral-7b-hf` |
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` |
| `llava-1.5-7b` | `llava-hf/llava-1.5-7b-hf` |



## Datasets

Set `--data-root` or `DATA_ROOT` to the parent dataset directory (default:
`./datasets`). Use `--data-path` to override an individual dataset's location.

| `--dataset` | Required data / source |
| --- | --- |
| `videomme` | Extracted videos; annotations from [Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME) |
| `mmbench` | `MMBench-Video.tsv` and [restored video files](https://huggingface.co/datasets/opencompass/MMBench-Video#how-to-get-video-data) |
| `arkitscenes` | `test.jsonl` and extracted `arkitscenes.zip` videos from [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) |
| `aime25` | `aime2025-I.jsonl` and `aime2025-II.jsonl` from [AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) |

Adjust `--num-prompts` and `--start-idx` to select number of prompts.



## Run experiments

We provide a sample script that runs AR, cosSim + UCBSpec, cosSim + SP2EC, (original) KnapSpec,
KnapSpec + UCBSpec, and KnapSpec + SP2EC on five MMBench-Video prompts with
Qwen2.5-VL-7B-Instruct:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/models/Qwen2.5-VL-7B-Instruct \
DATA_ROOT=/path/to/datasets \
bash run_sample_mmbench_qwen2_5_vl_7b.sh
```

To run comparison with other model, dataset, or method:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_experiments.py \
  --model qwen2.5-vl-7b \
  --model-path /path/to/models/Qwen2.5-VL-7B-Instruct \
  --dataset mmbench --data-root /path/to/datasets \
  --cossim --knapspec
```

- `--cossim`: AR, cosSim + UCBSpec, cosSim + SP2EC.
- `--knapspec`: AR, plain KnapSpec, KnapSpec + UCBSpec, KnapSpec + SP2EC.


### Parameters

| Argument | Default | Meaning |
| --- | --- | --- |
| `--gamma` | `4` | Maximum draft length |
| `--use-tree` / `--use_tree` | `true` | Tree verification; `false` generates a chain of draft |
| `--beta` | `0.3` | SP2EC exploration parameter |
| `--ucb-l` / `--ucb_l` | `10` | UCBSpec exploration parameter L |

For example: `--gamma 5 --use_tree false --beta 0.2 --ucb_l 10`.

Use `--output-dir` to change the results location and `--help` for all options.

Remark: Tree size and whether tree decoding is enabled can have a considerable, device-dependent impact:
a larger tree offers more candidate paths but also increases verification work
and memory use. A larger tree is not necessarily faster, and chain decoding may
perform better on some devices.



## Acknowledgments

We thank the authors of **[KnapSpec: Self-Speculative Decoding via Adaptive Layer
Selection as a Knapsack Problem](https://github.com/kaist-flexml-lab/knapspec)** for their work and for making their
implementation available. This implementation builds on their codebase.

Please also cite the original KnapSpec work when using this implementation:

```bibtex
@inproceedings{cha2026knapspec,
      title={KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem},
      author={Cha, Seongjin and Kim, Gyuwan and Han, Dongsu and Yang, Tao and Han, Insu},
      booktitle={Proceedings of the Forty-Third International Conference on Machine Learning (ICML)},
      year={2026}
}
```

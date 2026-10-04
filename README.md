# finetune_laya_model

Fine-tune [Laya](https://huggingface.co/convaiinnovations/laya), Convai Innovations' typed-decision model, on Apple Silicon (MPS) with **LoRA adapters**, using the [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) dataset.

## Overview

Laya is a 421M-parameter encoder (ModernBERT-large plus a small decision head). Given a **state** (text or JSON) and **typed questions**, it answers all of them in one forward pass and returns calibrated probabilities. It does not generate text.

| Question type | You provide | You get back |
|---|---|---|
| `choice` | named options with descriptions | a probability per option and the top pick |
| `score` | an ordered scale | a probability over the levels |
| `noul` | a yes/no proposition | one calibrated probability that it is true |

This repo contains one script, `laya_finetune_typed_decisions_mps_adapter.py`. It fine-tunes Laya on a MacBook by training small LoRA adapters instead of every weight, then exports both a normal Laya checkpoint and the adapter files.

## Files

| File | Purpose |
|---|---|
| `laya_finetune_typed_decisions_mps_adapter.py` | Fine-tuning with LoRA adapters (or head-only, or full), temperature calibration, and checkpoint/adapter export |

## Training modes

Choose a mode with `--mode`:

| Mode | What is trained | Notes |
|---|---|---|
| `lora` (default) | LoRA adapters inside the encoder, plus the decision head | The base encoder weights stay frozen. Lowest memory use of the options that change the encoder. |
| `head` | The decision head only | The encoder is fully frozen. Cheapest, but cannot change how the encoder reads text. |
| `full` | Every weight | The original behavior. Highest memory and disk use. |

### What LoRA does

LoRA (Low-Rank Adaptation) freezes a weight matrix `W` and learns a small correction built from two thin matrices, `B` and `A`:

```
W' = W + (alpha / r) * B * A
```

Only `A` and `B` are trained, so the trainable parameter count is a small fraction of the model, and the optimizer needs far less memory. In this script the adapters are attached to the ModernBERT attention and MLP layers (`Wqkv`, `Wo`, `Wi`). The decision head is always trained in every mode.

## Requirements

- A Mac with Apple Silicon and a **native arm64 Python** (3.10 or newer). An Intel Python fails with `Bad CPU type in executable`.
- About 15 to 20 GB of free disk space. The base checkpoint is roughly 2.4 GB, and macOS may use swap while training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
file .venv/bin/python        # should say arm64
pip install laya datasets peft
```

## Quick smoke test

Train on only 300 items first, to check that everything works:

```bash
python laya_finetune_typed_decisions_mps_adapter.py --model-dir ./laya_base --output-dir ./laya_lora_test --items ./train_items.pt --epochs 1 --micro-batch 1 --grad-accum 32 --max-items 300 --device mps
```

The log should start with a line like `Mode: lora; trainable parameters: ... of ... (...%)`. A small percentage means the adapters were attached and the base weights are frozen.

## Full run

Run it as a single line:

```bash
python laya_finetune_typed_decisions_mps_adapter.py --model-dir ./laya_base --output-dir ./laya_finetuned_typed_decisions --items ./train_items.pt --epochs 1 --micro-batch 1 --grad-accum 32 --device mps
```

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--model-dir` | `./laya_base` | Folder with the base checkpoint. If `model.safetensors` is missing, it is downloaded there. |
| `--output-dir` | `./laya_finetuned_typed_decisions` | Where the fine-tuned model is saved. |
| `--items` | `./train_items.pt` | Cache file for the preprocessed training data. |
| `--epochs` | 4 | Passes over the training data. |
| `--micro-batch` | 2 | Items per forward/backward pass. Lower uses less memory. |
| `--grad-accum` | 16 | Passes accumulated before each weight update. Effective batch = micro-batch x grad-accum. |
| `--calib-max` | 400 | Maximum items held out for temperature calibration. |
| `--device` | `auto` | `auto`, `mps`, or `cpu`. |
| `--mode` | `lora` | `lora`, `head`, or `full` (see above). |
| `--lora-r` | 16 | LoRA rank. Higher means more capacity and more trainable parameters. |
| `--lora-alpha` | 32 | LoRA scale. The correction is multiplied by `alpha / r`. |
| `--lora-dropout` | 0.05 | Dropout applied inside the LoRA layers during training. |
| `--lora-lr` | 2e-4 | Learning rate for the LoRA matrices. |
| `--head-lr` | 1e-4 | Learning rate for the decision head. |
| `--max-items` | 0 | Train on only the first N items (0 means all). Useful for smoke tests. |
| `--force-preprocess` | off | Rebuild the cached training items. |
| `--no-checkpointing` | off | Turn off gradient checkpointing (uses more memory). |
| `--model-id` | `convaiinnovations/laya` | Hugging Face repo to download the base checkpoint from. |

## What the script does

1. Downloads the base checkpoint (if needed) and the `LocalLLaMA/typed-decisions` training split (1,200 cases, 6,000 decisions), and caches the preprocessed items in `train_items.pt`.
2. Holds out part of the items for calibration, and trains on the rest.
3. Applies the chosen mode: injects LoRA adapters into the encoder and freezes the base weights (or freezes the whole encoder, or leaves everything trainable).
4. Trains with a reinforcement-learning loss (proper-scoring-rule rewards) plus a cross-entropy term.
5. Refits the confidence temperatures per question type on the held-out items.
6. Saves the final model and, in `lora` mode, the adapter files.

## Output

After a run, `--output-dir` contains:

| Path | Contents |
|---|---|
| `model.safetensors`, `rl_agent_config.json`, `encoder/`, `tokenizer/` | A normal Laya checkpoint. In `lora` mode the adapters are merged into the base weights, so it loads like any other Laya model. |
| `adapter/` (`lora` mode only) | `lora_adapter.safetensors` (the LoRA matrices), `head.safetensors` (the trained decision head), and `lora_config.json`. |
| `checkpoint_latest/` | Saved after each epoch. In `lora` mode it holds only the adapters and head (small), not a loadable model. |

The `adapter/` files are kept for inspection and reuse. The script itself does not yet load an adapter back in, so to use the model, load the merged checkpoint.

## Use the fine-tuned model

```python
import json
import laya

agent = laya.load("./laya_finetuned_typed_decisions")

state = {
    "subject": "Duplicate charge",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel.",
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, errors",
            "other": "everything else",
        },
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer explicitly threaten to cancel?",
    },
}

result = agent.predict(state, questions)
print(json.dumps(result["answers"], indent=2, default=str))
```

## Notes and limits

- LoRA changes the model less than full fine-tuning, so results may land slightly lower. Compare against the base model on held-out data before relying on it.
- The model is saved only after the training loop ends (the merged file is written once, at the end). A failure during that final save loses the run, so keep plenty of free disk space.
- The training data covers four workflows: agent-trace observability, customer service, invoice processing, and security incidents. Fine-tuning on it can change behavior on other kinds of text.
- The gold labels are soft targets produced by a teacher model, so they are noisy.

## Troubleshooting

- **`Bad CPU type in executable`**: a tool or Python is an Intel build. Use arm64 binaries (`/opt/homebrew/bin`) and recreate the venv with an arm64 Python.
- **`ModuleNotFoundError: No module named 'datasets'` or `'peft'`**: run `pip install datasets peft` in the active venv.
- **`No space left on device`**: free up disk space and rerun. The weights are only written at the end, so the run must be repeated.
- **`unrecognized arguments:`**: a stray space or a missing space in the command. Run it as one line, with a space between each option and its value.
- **Nothing is printed**: the script file is probably incomplete. It should be 495 lines and end with `main()`.
- **A file named `laya.py`** next to your scripts shadows the installed package and causes a circular import. Rename it.

## Large files

Model folders (`laya_base/`, `laya_finetuned_typed_decisions/`, `laya_lora_test/`) and `train_items.pt` are large and should not be committed. Keep them in `.gitignore`, and publish fine-tuned weights on Hugging Face if you want to share them.

## Credits

- Laya model and the original fine-tuning notebook: [Convai Innovations / NandhaKishorM/laya](https://github.com/NandhaKishorM/laya). This script is adapted from that notebook; check the upstream repository for its license before redistributing.
- Dataset: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).

# finetune_laya_model

Fine-tune and validate [Laya](https://huggingface.co/convaiinnovations/laya), Convai Innovations' typed-decision model, on Apple Silicon (MPS) using the [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) dataset.

## What this is

Laya is a 421M-parameter encoder (ModernBERT-large plus a small decision head). Given a **state** (text or JSON) and **typed questions**, it answers all of them in a single forward pass and returns calibrated probabilities. There is no text generation.

| Question type | You provide | You get back |
|---|---|---|
| `choice` | named options with descriptions | a probability per option and the top pick |
| `score` | an ordered scale | a probability over the levels |
| `noul` | a yes/no proposition | one calibrated probability that it is true |

This repo holds a training script that fine-tunes Laya on a MacBook, plus a small script to try a model locally.

## Files

| File | Purpose |
|---|---|
| `laya_finetune_typed_decisions_mps.py` | Full fine-tuning of Laya (encoder and head) on typed-decisions, followed by temperature calibration |
| `laya_local_agent.py` | Small example that loads Laya and answers typed questions about a sample state |

## Requirements

- A Mac with Apple Silicon and a **native arm64 Python** (3.10 or newer). An Intel Python fails with `Bad CPU type in executable`.
- About 15 to 20 GB of free disk space. The base checkpoint is roughly 2.4 GB, and macOS may use swap during training.
- Enough memory for full fine-tuning in fp32. A 16 GB machine is the practical minimum.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
file .venv/bin/python        # should say arm64
pip install laya datasets
```

## Fine-tune

Run it as a single line:

```bash
python laya_finetune_typed_decisions_mps.py --model-dir ./laya_base --output-dir ./laya_finetuned_typed_decisions --items ./train_items.pt --epochs 1 --micro-batch 1 --grad-accum 32 --calib-max 400 --device mps
```

| Argument | Meaning |
|---|---|
| `--model-dir` | Folder with the base checkpoint. If `model.safetensors` is missing, it is downloaded there. |
| `--output-dir` | Where the fine-tuned model is saved. |
| `--items` | Cache file for the preprocessed training data (`train_items.pt`). |
| `--epochs` | Passes over the training data. |
| `--micro-batch` | Items per forward/backward pass. Lower uses less memory. |
| `--grad-accum` | Passes accumulated before each weight update. Effective batch = micro-batch x grad-accum. |
| `--calib-max` | Maximum items held out for temperature calibration. |
| `--device` | `auto`, `mps`, or `cpu`. |
| `--force-preprocess` | Rebuild the cached training items. |
| `--no-checkpointing` | Turn off gradient checkpointing (uses more memory). |

### What the script does

1. Downloads the base checkpoint (if needed) and the `LocalLLaMA/typed-decisions` training split (1,200 cases, 6,000 decisions).
2. Packs each decision into model-ready items and caches them in `train_items.pt`.
3. Holds out items for calibration and trains on the rest. Both the encoder and the head are updated, using a reinforcement-learning loss with proper-scoring-rule rewards plus a cross-entropy term.
4. Refits the confidence temperatures per question type on the held-out items.
5. Saves `model.safetensors`, `rl_agent_config.json`, and the tokenizer to `--output-dir`.

The model is saved once per epoch, so a failure during the final save loses the whole epoch. Keep plenty of free disk space.

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

## Troubleshooting

- **`Bad CPU type in executable`**: a tool or Python is an Intel build. Use arm64 binaries (`/opt/homebrew/bin`) and recreate the venv with an arm64 Python.
- **`No space left on device`** during the final save: free up disk space and rerun. Training has to be repeated because the weights are only written at the end of the epoch.
- **`unrecognized arguments:`** with nothing listed: a stray space after a `\` in a multi-line command. Run it as one line.
- **`ModuleNotFoundError: No module named 'datasets'`**: run `pip install datasets` in the active venv.
- **A file named `laya.py`** next to your scripts shadows the installed package and causes a circular import. Rename it.

## Large files

Model folders (`laya_base/`, `laya_finetuned_typed_decisions/`) and `train_items.pt` are large and should not be committed. Keep them in `.gitignore`, and publish fine-tuned weights on Hugging Face if you want to share them.

## Credits

- Laya model and the original fine-tuning notebook: [Convai Innovations / NandhaKishorM/laya](https://github.com/NandhaKishorM/laya). Check the upstream repository for its license before redistributing.
- Dataset: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).

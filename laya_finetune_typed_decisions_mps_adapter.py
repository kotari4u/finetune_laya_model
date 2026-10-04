"""Laya typed-decisions fine-tuning on Apple Silicon, with LoRA adapter support.

Based on notebooks/laya_finetune_typed_decisions_mps.py from the Laya repository.
The only additions are the --mode options and the adapter export:

  --mode lora   Freeze the ModernBERT encoder and train small LoRA adapters inside it,
                plus the decision head (default).
  --mode head   Freeze the encoder completely and train only the decision head.
  --mode full   Original behaviour: train every weight.

Outputs (in --output-dir):
  model.safetensors, rl_agent_config.json, encoder/, tokenizer/
      A normal Laya checkpoint. In lora mode the adapters are merged into the base
      weights, so laya.load(output_dir) works exactly as before.
  adapter/   (lora mode only)
      lora_adapter.safetensors  the LoRA matrices only (a few MB)
      head.safetensors          the trained decision head
      lora_config.json          rank, alpha, dropout, target layers

Examples:
  python laya_finetune_typed_decisions_mps_adapter.py --model-dir ./laya_base --epochs 1 --micro-batch 1 --grad-accum 32
  python laya_finetune_typed_decisions_mps_adapter.py --mode head --epochs 1
  python laya_finetune_typed_decisions_mps_adapter.py --max-items 300 --epochs 1     # quick smoke test
"""

import argparse
import gc
import json
import math
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

MODEL_ID = "convaiinnovations/laya"
DATASET_ID = "LocalLLaMA/typed-decisions"
DEFAULT_MODEL_DIR = "./laya_base"
DEFAULT_ITEMS = "./train_items.pt"
DEFAULT_OUTPUT_DIR = "./laya_finetuned_typed_decisions"

# ModernBERT linear layers: attention (Wqkv, Wo) and MLP (Wi, Wo).
LORA_TARGETS = ["Wqkv", "Wo", "Wi"]


def choose_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS is unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def prepare_model(model_dir):
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists():
        print(f"Downloading {MODEL_ID} to {model_dir} ...")
        snapshot_download(MODEL_ID, local_dir=str(model_dir))
    _fix_tokenizer_config(str(model_dir))
    return str(model_dir)


def build_training_item(tokenizer, cfg, state, question, gold_question):
    qtype = question["type"]
    criteria = question.get("criteria", {})
    if qtype == "choice":
        keys = list(criteria.keys())
        target = [gold_question["probabilities"].get(k, 0.0) for k in keys]
    elif qtype == "noul":
        target = [
            gold_question["probabilities"].get("false", 0.5),
            gold_question["probabilities"].get("true", 0.5),
        ]
    elif qtype == "score":
        n_levels = len(criteria) if isinstance(criteria, list) else 4
        target = [gold_question["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    else:
        return None
    total = sum(target)
    if total > 0:
        target = [float(x) / total for x in target]
    else:
        target = [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    n_options = len(render_options({"t": qtype, "crit": criteria}))
    sequence, markers = build_sequence(
        tokenizer,
        state,
        {"t": qtype, "ins": question["instructions"], "crit": criteria},
        cfg["max_len"],
        cfg["head_max_len"],
    )
    if len(markers) != n_options:
        return None
    return {
        "ids": sequence,
        "markers": markers,
        "qtype": QTYPES[qtype],
        "target": target,
        "label": label,
    }


def prepare_items(model_dir, items_path, force=False):
    items_path = Path(items_path)
    model_path = Path(model_dir).resolve()
    with open(model_path / "rl_agent_config.json") as f:
        cfg = json.load(f)
    max_len = cfg.get("max_len", 1024)
    head_max_len = cfg.get("head_max_len", 256)
    cache_meta_path = items_path.with_name(items_path.name + ".meta.json")
    cache_key = {
        "model_dir": str(model_path),
        "max_len": max_len,
        "head_max_len": head_max_len,
        "dataset": DATASET_ID,
        "split": "train",
    }
    if items_path.exists() and not force:
        if not cache_meta_path.exists():
            # Legacy item files predate the sidecar metadata. They remain
            # usable offline; newly written caches always receive a key.
            print(f"Using legacy cached training items without metadata: {items_path}")
            return
        try:
            with open(cache_meta_path) as f:
                cached_key = json.load(f)
        except (OSError, json.JSONDecodeError):
            cached_key = None
        if cached_key == cache_key:
            print(f"Using cached training items: {items_path}")
            return
        print("Training-item cache key changed; rebuilding the cache.")

    # Keep datasets optional when a compatible local cache is already available.
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
    print(f"Downloading dataset {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, "all", split="train")
    items = []
    skipped = 0
    for row in dataset:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, question in questions.items():
            if qid not in gold:
                continue
            item = build_training_item(
                tokenizer,
                {**cfg, "max_len": max_len, "head_max_len": head_max_len},
                state,
                question,
                gold[qid],
            )
            if item is None:
                skipped += 1
            else:
                items.append(item)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, items_path)
    with open(cache_meta_path, "w") as f:
        json.dump(cache_key, f, indent=2)
    print(f"Saved {len(items)} training items to {items_path}; skipped={skipped}")


def collate(items, pad_id):
    batch_size = len(items)
    seq_len = max(len(item["ids"]) for item in items)
    kmax = max(len(item["markers"]) for item in items)
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention = torch.zeros((batch_size, seq_len), dtype=torch.long)
    marker_pos = torch.zeros((batch_size, kmax), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, kmax), dtype=torch.bool)
    target = torch.zeros((batch_size, kmax), dtype=torch.float32)
    for i, item in enumerate(items):
        length = len(item["ids"])
        input_ids[i, :length] = torch.tensor(item["ids"], dtype=torch.long)
        attention[i, :length] = 1
        k = len(item["markers"])
        marker_pos[i, :k] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[i, :k] = True
        target[i, :len(item["target"])] = torch.tensor(item["target"], dtype=torch.float32)
    return (
        input_ids,
        attention,
        marker_pos,
        marker_mask,
        target,
        torch.tensor([item["qtype"] for item in items], dtype=torch.long),
    )


def fit_temperature(samples):
    if len(samples) < 10:
        return 1.0
    kmax = max(len(logits) for logits, _ in samples)
    logits = torch.full((len(samples), kmax), -1e4)
    targets = torch.zeros((len(samples), kmax))
    for i, (values, target) in enumerate(samples):
        logits[i, :len(values)] = torch.as_tensor(values)
        targets[i, :len(target)] = torch.as_tensor(target, dtype=torch.float32)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_temperature.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), 0.1, 10.0).item())


# --------------------------------------------------------------------------- #
# Adapter support
# --------------------------------------------------------------------------- #
def apply_mode(model, args):
    """Freeze weights / inject LoRA according to --mode. Returns the LoraConfig or None."""
    lora_cfg = None
    if args.mode == "lora":
        from peft import LoraConfig, inject_adapter_in_model

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS,
            bias="none",
        )
        # In-place injection: model.encoder stays the original ModernBERT module
        # (no PeftModel wrapper), so the rest of the Laya code is unaffected.
        inject_adapter_in_model(lora_cfg, model.encoder)

    for name, param in model.named_parameters():
        if "encoder." in name:
            param.requires_grad = args.mode == "full" or "lora_" in name
        else:
            param.requires_grad = True  # decision head is always trained
    return lora_cfg


def export_state_dict(model):
    """fp16 CPU state dict with LoRA deltas merged into the base weights.

    The keys are the original Laya names (for example
    encoder.layers.0.attn.Wqkv.weight), so the result loads with strict=True.
    Without LoRA modules it is simply the normal state dict.
    """
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            base = module.base_layer
            weight = base.weight.detach().float()
            for adapter in module.lora_A.keys():
                a = module.lora_A[adapter].weight.detach().float()
                b = module.lora_B[adapter].weight.detach().float()
                weight = weight + module.scaling[adapter] * (b @ a)
            out[f"{name}.weight"] = weight.half().cpu().contiguous()
            if base.bias is not None:
                out[f"{name}.bias"] = base.bias.detach().half().cpu().contiguous()
    for key, value in model.state_dict().items():
        if "lora_" in key or ".base_layer." in key:
            continue
        out[key] = value.detach().half().cpu().contiguous()
    return out


def save_adapter(model, path, epoch):
    """Save only the LoRA matrices and the decision head (small files)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    lora = {k: v.detach().cpu().contiguous() for k, v in state.items() if "lora_" in k}
    head = {k: v.detach().half().cpu().contiguous() for k, v in state.items() if "encoder." not in k}
    save_file(lora, str(path / "lora_adapter.safetensors"))
    save_file(head, str(path / "head.safetensors"))
    with open(path / "lora_config.json", "w") as f:
        json.dump(
            {
                "base_model": MODEL_ID,
                "target_modules": LORA_TARGETS,
                "epoch": epoch,
            },
            f,
            indent=2,
        )


def save_checkpoint(model, tokenizer, cfg, output_dir, epoch, final=False):
    path = Path(output_dir) if final else Path(output_dir) / "checkpoint_latest"
    path.mkdir(parents=True, exist_ok=True)
    save_file(export_state_dict(model), str(path / "model.safetensors"))
    model.encoder.config.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "tokenizer")
    with open(path / "checkpoint_meta.json", "w") as f:
        json.dump({"epoch": epoch, "final": final}, f, indent=2)
    with open(path / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def train(args, model_dir, items_path, device):
    with open(Path(model_dir) / "rl_agent_config.json") as f:
        cfg = json.load(f)
    cfg.update({"max_tokens_per_batch": 2048, "max_len": 1024, "head_max_len": 256})
    if not args.no_checkpointing:
        cfg["gradient_checkpointing"] = True

    tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    model = build_model(cfg, encoder_dir=Path(model_dir) / "encoder")
    model.load_state_dict(load_file(str(Path(model_dir) / "model.safetensors")), strict=True)
    model.float()

    lora_cfg = apply_mode(model, args)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Mode: {args.mode}; trainable parameters: {n_trainable:,} of {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    if not args.no_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    model.to(device).train()

    all_items = torch.load(items_path, map_location="cpu", weights_only=False)
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(args.calib_max, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    if args.max_items:
        train_items = train_items[:args.max_items]

    encoder_params = [p for n, p in model.named_parameters() if "encoder." in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n and p.requires_grad]
    encoder_lr = args.lora_lr if args.mode == "lora" else 2.5e-5
    groups = [{"params": head_params, "lr": args.head_lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    updates = max(1, math.ceil(len(train_items) / args.micro_batch / args.grad_accum) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=1e-6
    )

    print(f"Device: {device}")
    print(f"Training items: {len(train_items)}; calibration items: {len(calib_items)}")
    print(f"micro_batch={args.micro_batch}; grad_accum={args.grad_accum}; epochs={args.epochs}")

    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(train_items)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_batches = 0
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, args.epochs - 1)
        for start in range(0, len(train_items), args.micro_batch):
            chunk = train_items[start:start + args.micro_batch]
            ids, attention, positions, mask, target, qtype = collate(chunk, tokenizer.pad_token_id)
            ids, attention = ids.to(device), attention.to(device)
            positions = positions.to(device)
            mask = mask.to(device)
            target = target.to(device)
            qtype = qtype.to(device)
            logits, activation = model(ids, attention, positions, mask, qtype)
            logits = logits.float()
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            noisy_logits = logits.detach().unsqueeze(0) + eps
            probabilities = torch.softmax(noisy_logits.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(
                    probabilities,
                    target.unsqueeze(0),
                    qtype,
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            logp = -(((noisy_logits - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(
                target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (loss_rl + loss_ce + 0.0 * activation.sum()) / args.grad_accum
            loss.backward()
            n_batches += 1
            if n_batches % args.grad_accum == 0 or start + args.micro_batch >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * args.grad_accum
            if n_batches % 100 == 0:
                print(f"epoch {epoch + 1}/{args.epochs}, step {n_batches}, loss={loss.item() * args.grad_accum:.4f}")
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1}/{args.epochs} complete; avg_loss={avg_loss:.4f}")
        if args.mode == "lora":
            # Small per-epoch checkpoint: adapters + head only (the full model is written at the end).
            save_adapter(model, Path(args.output_dir) / "checkpoint_latest", epoch + 1)
        else:
            save_checkpoint(model, tokenizer, cfg, args.output_dir, epoch + 1)

    print("Running temperature calibration ...")
    model.eval()
    samples = [[] for _ in range(3)]
    with torch.no_grad():
        for start in range(0, len(calib_items), args.micro_batch):
            chunk = calib_items[start:start + args.micro_batch]
            ids, attention, positions, mask, target, qtype = collate(chunk, tokenizer.pad_token_id)
            logits, _ = model(
                ids.to(device), attention.to(device), positions.to(device), mask.to(device), qtype.to(device)
            )
            for i, item in enumerate(chunk):
                qtype_id = item["qtype"]
                samples[qtype_id].append(
                    (logits[i, :len(item["markers"])].cpu(), item["target"])
                )
    temperatures = [fit_temperature(group) if group else 1.2 for group in samples]
    save_checkpoint(model, tokenizer, cfg, args.output_dir, args.epochs, final=True)
    if lora_cfg is not None:
        save_adapter(model, Path(args.output_dir) / "adapter", args.epochs)
    cfg.update({
        "fine_tuned": True,
        "model_name": "laya-typed-decisions",
        "temperature": temperatures,
    })
    cfg.pop("temperature_by_options", None)
    with open(Path(args.output_dir) / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    (Path(args.output_dir) / "checkpoint_latest").mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "checkpoint_latest" / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Model saved to {args.output_dir}")
    print(f"Temperatures: {temperatures}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on Apple Silicon MPS (full, head-only, or LoRA adapters)")
    parser.add_argument("model_dir", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--model-dir", dest="model_dir_option")
    parser.add_argument("--output-dir", dest="output_dir_option")
    parser.add_argument("--items", default=DEFAULT_ITEMS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--calib-max", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    # Adapter options
    parser.add_argument("--mode", choices=["lora", "head", "full"], default="lora",
                        help="lora: LoRA adapters + head; head: head only; full: all weights")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling (alpha / r is the multiplier)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=2e-4, help="learning rate for the LoRA matrices")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="learning rate for the decision head")
    parser.add_argument("--max-items", type=int, default=0,
                        help="train on only the first N items (0 = all); useful as a quick smoke test")
    args = parser.parse_args()

    if args.model_id != MODEL_ID:
        # Keep the requested model ID local to preparation by updating the
        # module constant before prepare_model() is called.
        globals()["MODEL_ID"] = args.model_id

    args.model_dir = args.model_dir_option or args.model_dir or DEFAULT_MODEL_DIR
    args.output_dir = args.output_dir_option or args.output_dir or DEFAULT_OUTPUT_DIR
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    model_dir = prepare_model(args.model_dir)
    prepare_items(model_dir, args.items, force=args.force_preprocess)
    train(args, model_dir, args.items, device)
    gc.collect()


if __name__ == "__main__":
    main()
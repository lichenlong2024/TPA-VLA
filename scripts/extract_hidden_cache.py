"""Extract VLM hidden-state caches from a JSONL manifest.

This utility connects TPA-VLA's component training scripts to a standard VLM
stack. Each manifest row should contain an image path, an instruction, a
proprio vector, and an action chunk.

Example JSONL row:
{"image": "frames/000001.png", "instruction": "put the mug in the microwave",
 "proprio": [0.0, ...], "actions": [[0.1, ...], ...]}

Use `--lora_adapter_path` when extracting Phase-I adapted-backbone hidden
states. Omit it when extracting restored frozen-backbone hidden states for
Phase II.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import torch
from PIL import Image
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return rows


def load_model(model_path: str, lora_adapter_path: str, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if lora_adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install `peft` to use --lora_adapter_path.") from exc
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.to(device).eval()
    return processor, model


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def fit_sequence_length(hidden: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Pad or crop [layers, tokens, hidden_dim] to a fixed token length."""
    if hidden.shape[1] == sequence_length:
        return hidden
    if hidden.shape[1] > sequence_length:
        return hidden[:, :sequence_length, :]
    pad = torch.zeros(
        hidden.shape[0],
        sequence_length - hidden.shape[1],
        hidden.shape[2],
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return torch.cat([hidden, pad], dim=1)


def image_inputs(row: Dict[str, Any], image_root: Path, image_key: str) -> Union[Image.Image, List[Image.Image]]:
    value = row[image_key]
    if isinstance(value, list):
        return [Image.open(resolve_path(image_root, item)).convert("RGB") for item in value]
    return Image.open(resolve_path(image_root, value)).convert("RGB")


@torch.inference_mode()
def extract_one(processor, model, row: Dict[str, Any], image_root: Path, args, device: torch.device) -> torch.Tensor:
    image = image_inputs(row, image_root, args.image_key)
    text = row[args.instruction_key]
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output did not expose `hidden_states`; add a model-specific hook for this backbone.")
    hidden = torch.stack([h[0].detach().float().cpu() for h in hidden_states], dim=0)
    return fit_sequence_length(hidden, args.sequence_length)


def tensorize(rows: Iterable[Dict[str, Any]], key: str, dtype: torch.dtype) -> torch.Tensor:
    values = [row[key] for row in rows]
    return torch.tensor(values, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL trajectory/frame manifest.")
    parser.add_argument("--image_root", default="", help="Root used for relative image paths.")
    parser.add_argument("--model_path", required=True, help="HF-compatible VLM path.")
    parser.add_argument("--lora_adapter_path", default="", help="Optional LoRA adapter for Phase-I hidden states.")
    parser.add_argument("--output", required=True, help="Output .pt cache path.")
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--instruction_key", default="instruction")
    parser.add_argument("--proprio_key", default="proprio")
    parser.add_argument("--actions_key", default="actions")
    parser.add_argument("--sequence_length", type=int, default=519)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(Path(args.manifest))
    image_root = Path(args.image_root) if args.image_root else Path(args.manifest).parent
    processor, model = load_model(args.model_path, args.lora_adapter_path, dtype, device)

    hidden = []
    for row in tqdm(rows, dynamic_ncols=True, desc="extract-hidden-cache"):
        hidden.append(extract_one(processor, model, row, image_root, args, device))

    payload = {
        "hidden_states": torch.stack(hidden, dim=0),
        "proprio": tensorize(rows, args.proprio_key, torch.float32),
        "actions": tensorize(rows, args.actions_key, torch.float32),
        "metadata": {
            "model_path": args.model_path,
            "uses_lora_adapter": bool(args.lora_adapter_path),
            "sequence_length": args.sequence_length,
            "num_samples": len(rows),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Wrote hidden-state cache: {out}")


if __name__ == "__main__":
    main()

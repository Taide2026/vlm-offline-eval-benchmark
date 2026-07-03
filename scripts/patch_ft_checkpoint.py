"""Backfill KV-shared-layer weights missing from finetuned Gemma checkpoints.

HF transformers does not instantiate ``k_proj``/``v_proj``/``k_norm`` on
Gemma's KV-shared layers (they reuse an earlier layer's KV cache), so
``save_pretrained`` after a full finetune omits those tensors. The Google base
checkpoints ship them anyway, and vLLM's strict weight loader requires them.
This script copies the missing tensors from the base model into the finetuned
checkpoint's cached ``model.safetensors``, in place.

Usage:
    uv run python scripts/patch_ft_checkpoint.py <ft_model_id> <base_model_id>

Example:
    uv run python scripts/patch_ft_checkpoint.py \
        THChou1220/gemma-4-e2b-kinetics54K_FFT google/gemma-4-E2B-it

Note: this edits the local HF cache copy. Re-downloading the model reverts
the patch; upload the patched file to the Hub to fix it permanently.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def patch(ft_model_id: str, base_model_id: str) -> int:
    """Copy tensors present in the base checkpoint but absent from the FT one.

    Args:
        ft_model_id: HuggingFace ID of the finetuned model to patch.
        base_model_id: HuggingFace ID of the base model to copy tensors from.

    Returns:
        Process exit code (0 on success).
    """
    ft_file = Path(snapshot_download(ft_model_id, allow_patterns=["*.safetensors"])) / "model.safetensors"
    base_file = Path(snapshot_download(base_model_id, allow_patterns=["*.safetensors"])) / "model.safetensors"

    ft = load_file(str(ft_file))
    base = load_file(str(base_file))
    missing = sorted(set(base) - set(ft))
    if not missing:
        print(f"Nothing to do: {ft_model_id} already has every tensor in {base_model_id}.")
        return 0

    print(f"Backfilling {len(missing)} tensors from {base_model_id}:")
    for key in missing:
        print(f"  + {key}")
        ft[key] = base[key]

    tmp = ft_file.with_suffix(".safetensors.patched")
    save_file(ft, str(tmp), metadata={"format": "pt"})
    tmp.replace(ft_file)
    print(f"Patched {ft_file}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(patch(sys.argv[1], sys.argv[2]))

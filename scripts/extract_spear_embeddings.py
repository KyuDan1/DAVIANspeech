"""Cache layer-wise SPEAR v2 embeddings for downstream detector heads.

SPEAR is jointly pretrained on speech and general audio.  We pool every
Zipformer stack rather than assuming that its final layer is optimal for
generation-artifact detection, which is different from its pretraining task.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import find_audio_files, load_audio  # noqa: E402
from presence import extract_segment, segment_starts  # noqa: E402


def load_spear(model_dir: Path, device: torch.device):
    """Load the repository-local custom model without a HF module cache."""
    package = "davianspeech_spear"
    package_spec = importlib.util.spec_from_file_location(
        package, model_dir / "__init__.py",
        submodule_search_locations=[str(model_dir)],
    )
    module = importlib.util.module_from_spec(package_spec)
    sys.modules[package] = module
    model_spec = importlib.util.spec_from_file_location(
        f"{package}.modeling_spear", model_dir / "modeling_spear.py"
    )
    modeling = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = modeling
    model_spec.loader.exec_module(modeling)
    return modeling.SpearModel.from_pretrained(model_dir).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models/spear-xlarge-speech-audio-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=160_000)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    files = find_audio_files(args.test_dir)[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    model = load_spear(args.model_dir, device)
    ids, vectors = [], []
    for path in tqdm(files, desc="SPEAR embeddings"):
        audio = load_audio(path)
        starts = segment_starts(len(audio), args.window)
        if args.max_windows and len(starts) > args.max_windows:
            indices = np.linspace(0, len(starts) - 1, args.max_windows, dtype=int)
            starts = [starts[index] for index in np.unique(indices)]
        per_window = []
        for start in starts:
            waveform = torch.from_numpy(
                extract_segment(audio, start, args.window)
            ).unsqueeze(0).to(device)
            lengths = torch.tensor([waveform.shape[1]], device=device)
            with torch.inference_mode():
                output = model(waveform, lengths)
            # Each hidden state is (batch, frames, 1280). Layer-wise pooling
            # preserves shallow acoustic and deep semantic representations.
            pooled = torch.cat(
                [hidden.float().mean(dim=1) for hidden in output["hidden_states"]],
                dim=-1,
            )
            per_window.append(pooled.cpu().numpy()[0])
        ids.append(path.stem)
        vectors.append(np.mean(per_window, axis=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, ids=np.asarray(ids), embeddings=np.asarray(vectors, dtype=np.float32)
    )
    print(f"Saved {len(ids)} SPEAR embeddings to {args.output}")


if __name__ == "__main__":
    main()

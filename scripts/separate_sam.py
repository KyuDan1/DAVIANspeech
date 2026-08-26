"""SAM-Audio separation pass: writes voice/music stems for the detector pass.

Run this with the `samaudio` environment, not the pipeline's. SAM-Audio wants
torchcodec, perception-models and friends, which cannot coexist with the
detector stack, so separation is a standalone pass that leaves 16 kHz stems on
disk for `pipeline.py --separator precomputed --stems-dir ...`.

    python scripts/separate_sam.py --test-dir data/test --out-dir stems/sam-large
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

TARGET_SR = 16_000
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}

# SAM-Audio's prompts work best as lowercase noun/verb phrases, per its README.
DEFAULT_PROMPT = "a person speaking or singing"


def to_mono_16k(waveform: torch.Tensor, source_sr: int) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform[None]
    mono = waveform.mean(0, keepdim=True).float()
    if source_sr != TARGET_SR:
        mono = torchaudio.functional.resample(mono, source_sr, TARGET_SR)
    return mono[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="facebook/sam-audio-large")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reranking-candidates", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from sam_audio import SAMAudio, SAMAudioProcessor

    files = sorted(
        (p for p in args.test_dir.iterdir()
         if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda p: p.stem,
    )
    if args.limit:
        files = files[:args.limit]
    if args.num_shards > 1:
        files = files[args.shard_index::args.num_shards]
    if not files:
        raise SystemExit(f"No audio under {args.test_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = SAMAudio.from_pretrained(args.checkpoint).eval().to(args.device)
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint)
    source_sr = processor.audio_sampling_rate
    print(f"{args.checkpoint} loaded, processor rate {source_sr} Hz", flush=True)

    for path in tqdm(files, desc="separate"):
        voice_path = args.out_dir / f"{path.stem}_voice.wav"
        music_path = args.out_dir / f"{path.stem}_music.wav"
        if not args.overwrite and voice_path.is_file() and music_path.is_file():
            continue

        batch = processor(audios=[str(path)], descriptions=[args.prompt]).to(args.device)
        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=False,
                reranking_candidates=args.reranking_candidates,
            )

        # The residual is everything the voice prompt did not claim, which is
        # the counterpart of Demucs' summed non-vocal stems.
        target = torch.as_tensor(result.target).detach().cpu()
        residual = torch.as_tensor(result.residual).detach().cpu()
        if target.dim() == 3:      # (batch, channels, samples)
            target, residual = target[0], residual[0]

        sf.write(voice_path, to_mono_16k(target, source_sr).numpy(), TARGET_SR)
        sf.write(music_path, to_mono_16k(residual, source_sr).numpy(), TARGET_SR)

    print(f"wrote stems for {len(files)} files to {args.out_dir}")


if __name__ == "__main__":
    main()

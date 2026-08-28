"""Generate Korean fakes with Audio8-TTS (DualAR codec-LLM), cloning each real speaker.

Each fake reuses its source utterance both as the speaker reference and as the
target text, so a real/fake pair differs only in whether the waveform was
generated -- not in speaker, not in wording. Output lands in
``<pool>/fake/audio8/<ID>.wav`` at 16 kHz, ready for build_eval_korean.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

MODEL_ID = "Audio8/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
MODEL_SR = 44_100
TARGET_SR = 16_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    # 1024 truncated mid-sentence on ~13 s inputs; the pool reaches 28 s.
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    out_dir = args.pool / "fake" / "audio8"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.pool / "meta.json").read_text("utf-8"))
    if args.limit:
        meta = meta[:args.limit]

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=torch.bfloat16
    ).eval().to("cuda")
    torch.manual_seed(args.seed)

    done = failed = 0
    start = time.time()
    for item in meta:
        destination = out_dir / f"{item['id']}.wav"
        if destination.exists():
            continue
        reference = args.pool / "real" / f"{item['id']}.wav"
        try:
            inputs = processor(
                text=[item["text"]],
                reference_audio=[str(reference)],
                reference_text=[item["text"]],
                return_tensors="pt",
            )
            with torch.inference_mode():
                generated = model.generate(
                    **{k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()},
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.8, top_p=0.95, top_k=50,
                    do_sample=True, return_dict_in_generate=True,
                )
                waveforms, lengths = model.decode_audio(generated.codes)
            audio = waveforms[0][:lengths[0]].float().cpu().numpy()
            if audio.size < MODEL_SR // 2 or audio.std() < 1e-4:
                # Empty or DC output; a silent clip would be a free win for the
                # detector and does not belong in the eval set.
                failed += 1
                print(f"  skip {item['id']}: degenerate (std={audio.std():.5f})", flush=True)
                continue
            resampled = librosa.resample(
                audio, orig_sr=MODEL_SR, target_sr=TARGET_SR, res_type="soxr_hq"
            )
            sf.write(destination, resampled.astype(np.float32), TARGET_SR)
            done += 1
            if done % 20 == 0:
                print(f"  {done} done, {time.time() - start:.0f}s", flush=True)
        except Exception as error:                      # noqa: BLE001
            failed += 1
            print(f"  skip {item['id']}: {type(error).__name__} {error}", flush=True)

    print(f"audio8: wrote {done}, skipped {failed}, {time.time() - start:.0f}s total")


if __name__ == "__main__":
    main()

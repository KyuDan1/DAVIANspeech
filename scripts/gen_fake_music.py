"""Generate INSTRUMENTAL AI music, so the two missing truth cells can be built.

Our eval set covers real-voice/real-music and fake-voice/real-music, but not the
two cells where the music itself is generated, because every fake-music source we
hold (SONICS, an AI pop set) carries singing -- laid under speech it makes
VOICE_FAKE ambiguous. Text-to-music fills the gap: prompts are instrument-only,
and the output is checked with PANNs so anything with a voice in it is dropped.

Do NOT substitute HTDemucs-stripped AI songs. Real beds reach the detector with
one separation pass; stripped fake beds would carry two, and the detector would
learn "separation residue = fake music" -- the same confound already distorting
the voice branch, installed deliberately as the music label.

MusicGen runs through transformers here on purpose: the audiocraft package pins
torch 2.1 / torchvision 0.16 and would take this environment apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Instrument-only prompts; nothing that would invite a vocal line.
STYLES = [
    "solo piano ballad, gentle, no vocals",
    "acoustic guitar fingerpicking, warm, instrumental",
    "lo-fi hip hop beat, mellow drums and rhodes, instrumental",
    "string quartet, classical, expressive",
    "ambient synth pad, slow, atmospheric",
    "upbeat funk groove, bass and drums, instrumental",
    "jazz trio, brushed drums, upright bass, piano",
    "electronic dance track, four on the floor, synth lead",
    "orchestral film score, strings and brass, dramatic",
    "reggae rhythm guitar and organ, instrumental",
    "country slide guitar, laid back, instrumental",
    "rock instrumental, distorted electric guitar riff",
    "bossa nova nylon guitar and light percussion",
    "chiptune melody, retro video game, instrumental",
    "marimba and hand percussion, tropical, instrumental",
    "cinematic tension, low strings and timpani",
]

VOICE_LABELS = ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
                "Singing", "Male singing", "Female singing", "Child singing",
                "Choir", "Rapping", "Vocal music", "Chant", "Humming"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--model", default="facebook/musicgen-medium")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--panns-dir", type=Path, default=None,
                        help="If given, drop any clip PANNs hears a voice in.")
    parser.add_argument("--voice-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = MusicgenForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16
    ).to("cuda").eval()
    rate = model.config.audio_encoder.sampling_rate
    # MusicGen emits 50 audio tokens per second.
    max_new_tokens = int(args.seconds * 50)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    checker = None
    if args.panns_dir:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from presence import PannsPresence
        checker = PannsPresence(args.panns_dir, device="cuda")
        # Reuse the presence model but ask it specifically about voices.
        from panns_inference import labels
        index = {name: i for i, name in enumerate(labels)}
        checker.voice_indices = [index[name] for name in VOICE_LABELS if name in index]

    written, rejected, index = 0, 0, 0
    meta = []
    while written < args.count:
        prompts = [STYLES[int(rng.integers(len(STYLES)))]
                   for _ in range(min(args.batch_size, args.count - written))]
        inputs = processor(text=prompts, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                                   max_new_tokens=max_new_tokens)
        for prompt, wave in zip(prompts, audio):
            clip = wave[0].float().cpu().numpy()
            clip = clip / max(np.abs(clip).max(), 1e-9) * 0.9
            if checker is not None:
                import librosa
                sixteen = librosa.resample(clip, orig_sr=rate, target_sr=16000,
                                           res_type="soxr_hq")
                voice, _ = checker.predict(sixteen.astype(np.float32))
                if voice > args.voice_threshold:
                    rejected += 1
                    index += 1
                    continue
            name = f"mg_{index:04d}.wav"
            sf.write(args.out_dir / name, clip.astype(np.float32), rate)
            meta.append({"file": name, "prompt": prompt, "sample_rate": rate,
                         "seconds": round(len(clip) / rate, 2), "model": args.model})
            written += 1
            index += 1
        print(f"  {written}/{args.count} written, {rejected} rejected", flush=True)

    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"instrumental fakes: {written} written, {rejected} rejected for voice content")


if __name__ == "__main__":
    main()

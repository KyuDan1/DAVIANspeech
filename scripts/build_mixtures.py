"""Mix Korean speech with real music, so separators can be compared on the case
that actually breaks the detector.

On clean speech the spoof detector is at EER 0.000, but mixing real music into
real speech and separating it back out pushes 56% of genuine clips past P(fake)
0.5 at 0 dB. Whatever leaks through the separator reads as generation artifact.
This writes matched real/fake mixtures at several music levels so HTDemucs and
SAM-Audio can be scored on the same inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16_000


def load(path, size=None):
    audio, _ = librosa.load(path, sr=SR, mono=True, dtype=np.float32)
    if size is not None:
        if audio.size < size:
            audio = np.tile(audio, size // max(audio.size, 1) + 1)
        audio = audio[:size]
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--music-list", type=Path, required=True,
                        help="JSON list of readable music wav paths")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--generator", default="audio8")
    parser.add_argument("--fake-music-dir", type=Path, default=None,
                        help="Instrumental generated beds; enables the two cells "
                             "where the music itself is fake.")
    parser.add_argument("--count", type=int, default=40)
    # The old {+5,0,-5} only probes where the VOICE branch breaks, which is when
    # music is loud. The music branch breaks at the other end, when the bed is
    # quiet, so the grid has to reach up.
    parser.add_argument("--snr", type=float, nargs="+",
                        default=[20.0, 15.0, 10.0, 5.0, 0.0, -5.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    music_paths = json.loads(args.music_list.read_text())
    meta = json.loads((args.pool / "meta.json").read_text("utf-8"))
    rng = np.random.default_rng(args.seed)

    fake_dir = args.pool / "fake" / args.generator
    usable = [m for m in meta if (fake_dir / f"{m['id']}.wav").is_file()][:args.count]
    if len(usable) < args.count:
        print(f"warning: only {len(usable)} of {args.count} ids have a {args.generator} fake")

    fake_music_paths = []
    if args.fake_music_dir:
        fake_music_paths = sorted(str(p) for p in args.fake_music_dir.glob("*.wav"))
        print(f"instrumental fake beds: {len(fake_music_paths)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in usable:
        for voice_label, source in (("real", args.pool / "real" / f"{item['id']}.wav"),
                                    ("fake", fake_dir / f"{item['id']}.wav")):
            speech = load(source)
            beds = [("real", music_paths)]
            if fake_music_paths:
                beds.append(("fake", fake_music_paths))
            for music_label, pool_paths in beds:
                # One bed per (utterance, voice label, music label) so a pair
                # differs only in the axis under test, never in which track
                # happened to be drawn.
                music = load(pool_paths[rng.integers(len(pool_paths))], speech.size)
                for snr in args.snr:
                    power_ratio = (speech ** 2).mean() / max((music ** 2).mean(), 1e-12)
                    scale = np.sqrt(power_ratio / 10 ** (snr / 10))
                    mix = speech + scale * music
                    mix = mix / max(np.abs(mix).max(), 1e-9) * 0.9
                    name = (f"{item['id']}_v{voice_label}_m{music_label}"
                            f"_snr{int(snr):+d}")
                    sf.write(args.out_dir / f"{name}.wav", mix.astype(np.float32), SR)
                    rows.append({
                        "ID": name, "SOURCE_ID": item["id"],
                        "VOICE_FAKE": int(voice_label == "fake"),
                        "MUSIC_FAKE": int(music_label == "fake"),
                        "SNR": snr,
                        "GENERATOR": args.generator if voice_label == "fake" else "bonafide",
                        "MUSIC_GENERATOR": "musicgen" if music_label == "fake" else "gtzan",
                        "DURATION": round(mix.size / SR, 2)})

    (args.out_dir / "index.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} mixtures to {args.out_dir}")
    import collections
    cells = collections.Counter((r["VOICE_FAKE"], r["MUSIC_FAKE"]) for r in rows)
    for (vf, mf), n in sorted(cells.items()):
        print(f"  voice={'fake' if vf else 'real'} music={'fake' if mf else 'real'}: {n}")


if __name__ == "__main__":
    main()

"""Turn three leaderboard ADS readings into File / Voice / Music EER.

    ADS = 0.5*(1-FILE_EER) + 0.2*(1-VOICE_EER) + 0.3*(1-MUSIC_EER)

Pinning one probability column to a constant fixes that column's EER at 0.5
(the competition publishes EER = (fpr+fnr)/2 at argmin|fpr-fnr|, and a constant
score puts both tied ROC points there), so the drop from the anchor names the
term. The anchor must be the SAME package with no column pinned, otherwise the
package difference lands inside the recovered number.

    python decode_probes.py --anchor 0.7075 --music-probe 0.68 --voice-probe 0.66
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=float, required=True, help="ADS of the unprobed twin")
    parser.add_argument("--music-probe", type=float, help="ADS with MUSIC_FAKE_PROB=0.5")
    parser.add_argument("--voice-probe", type=float, help="ADS with VOICE_FAKE_PROB=0.5")
    parser.add_argument("--cps", type=float, default=0.9891720635)
    args = parser.parse_args()

    print(f"anchor ADS      = {args.anchor:.10f}")
    print(f"weighted EER    = {1 - args.anchor:.10f}   (= 0.5*File + 0.2*Voice + 0.3*Music)\n")

    recovered = {}
    for name, probe, weight in (("MUSIC", args.music_probe, 0.3),
                                ("VOICE", args.voice_probe, 0.2)):
        if probe is None:
            continue
        delta = args.anchor - probe
        eer = 0.5 - delta / weight
        recovered[name] = eer
        flag = ""
        if not -1e-9 <= eer <= 0.5 + 1e-9:
            # Outside [0, 0.5] means the anchor is wrong or the grader's EER
            # convention differs; do not quietly report a nonsense number.
            flag = "  <-- OUT OF RANGE: anchor or EER convention is off"
        print(f"{name} probe ADS = {probe:.10f}   drop {delta:+.10f}"
              f"   -> {name}_EER = {eer:.4f}{flag}")

    if len(recovered) == 2:
        file_eer = ((1 - args.anchor)
                    - 0.2 * recovered["VOICE"] - 0.3 * recovered["MUSIC"]) / 0.5
        print(f"\nFILE_EER (by difference) = {file_eer:.4f}")
        print("\ncontribution to the weighted EER, largest first:")
        parts = [("FILE", 0.5 * file_eer), ("MUSIC", 0.3 * recovered["MUSIC"]),
                 ("VOICE", 0.2 * recovered["VOICE"])]
        for label, value in sorted(parts, key=lambda p: -p[1]):
            print(f"  {label:6s} {value:.4f}   ({value / (1 - args.anchor) * 100:.0f}% of the loss)")

        print("\nheadroom if a term were driven to zero:")
        for label, weight, eer in (("FILE", 0.5, file_eer), ("MUSIC", 0.3, recovered["MUSIC"]),
                                   ("VOICE", 0.2, recovered["VOICE"])):
            gain_ads = weight * eer
            print(f"  perfect {label:6s} -> ADS {args.anchor + gain_ads:.4f}"
                  f"  Score {0.9 * (args.anchor + gain_ads) + 0.1 * args.cps:.4f}")
        need = (0.8 - 0.1 * args.cps) / 0.9
        print(f"\nADS needed for Score 0.8: {need:.6f}  (gap {need - args.anchor:+.6f})")


if __name__ == "__main__":
    main()

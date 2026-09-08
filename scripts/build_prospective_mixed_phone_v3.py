#!/usr/bin/env python3
"""Reserve and build a leakage-audited mixed/telephone holdout.

This program never runs a detector.  ``reserve`` reads protocol metadata only
and creates 240 immutable base recipes.  ``build`` renders each base under one
clean and four paired telephone conditions, producing 1,200 files.

Strict mode is the default.  It requires the official ASVspoof2019 LA protocol
so every voice has an attack and speaker identity.  The deliberately weaker
source-disjoint/generator-seen mode is available only through the explicit
``--allow-generator-seen-fallback`` flag and is stamped into provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telephone_channel import apply_channel  # noqa: E402


SR = 16_000
MODES = ("concurrent", "partial_overlap", "sequential")
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
CHANNELS = (
    "clean", "g711_ulaw", "g722_wb", "opus_nb_8k",
    "transcode_g711_opus",
)
MUSIC_VOCAL_SCREEN_VERSION = "musiccaps_instrumental_text_v1"
MUSIC_VOCAL_SCREEN_PASS = (
    "musiccaps_explicit_instrumental_or_no_vocal_no_positive_voice_v1"
)
MUSIC_ACOUSTIC_SCREEN_PASS = (
    "musiccaps_no_positive_voice_text_panns_demucs_pass_v1"
)
FAKEMUSICCAPS_GENERATORS = (
    "MusicGen_medium", "audioldm2", "musicldm", "mustango",
    "stable_audio_open",
)
# The screen is deliberately conservative.  A MusicCaps caption/aspect must
# explicitly call the clip instrumental or state that voices/vocals are absent.
# Any remaining positive voice evidence rejects the source group.
EXPLICIT_INSTRUMENTAL_PATTERNS = (
    r"\binstrumental(?:\s+(?:music|track|piece|song|recording))?\b",
    r"\bno\s+(?:human\s+)?(?:voice|voices|vocal|vocals)\b",
    r"\bwithout\s+(?:any\s+)?(?:voice|voices|vocal|vocals)\b",
    r"\b(?:does|do)\s+not\s+(?:contain|include|feature|have)\s+"
    r"(?:any\s+)?(?:voice|voices|vocal|vocals)\b",
)
POSITIVE_VOICE_PATTERN = (
    r"\b(?:vocal(?:s|ist|ists)?|singer(?:s)?|singing|sings|sang|sung|speech|"
    r"spoken|talk(?:s|er|ers|ing)?|rap(?:s|per|pers|ping)?|choir|choral|"
    r"chant(?:s|er|ers|ing)?|humm(?:ed|ing)?|voice(?:s)?|voice[- ]?over|"
    r"lyrics?|narrat(?:e|ed|es|ing|ion|or)|a\s*cappella|acapella)\b"
)
IDENTITY_COLUMNS = (
    "ID", "GROUP_ID", "VOICE_SOURCE_ID", "MUSIC_SOURCE_ID", "SOURCE_FILE",
    "FIRST_SOURCE_ID", "SECOND_SOURCE_ID", "FIRST_GROUP", "SECOND_GROUP",
    "MUSIC_GROUP", "MUSIC_GROUP_ID", "FMA_TRACK_ID", "PAIR_GROUP",
    "PARENT_ID", "VOICE_SPEAKER",
)
MUSIC_IDENTITY_COLUMNS = (
    "GROUP_ID", "MUSIC_SOURCE_ID", "MUSIC_GROUP", "MUSIC_GROUP_ID",
    "FMA_TRACK_ID", "SOURCE_FILE",
)
PREDICTION_COLUMNS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)
MIXFAKE_LICENSES = (
    {
        "source": "MixFake",
        "url": "https://huggingface.co/datasets/Tnxts/MixFake",
        "license": "CC-BY-4.0 (repository declaration; verify redistribution terms)",
    },
    {
        "source": "ASVspoof2019 LA foreground",
        "url": "https://www.asvspoof.org/index2019.html",
        "license": "ASVspoof terms; verify for the intended use",
    },
    {
        "source": "FMA / FakeMusicCaps / SONICS backgrounds",
        "url": "recorded by MixFake protocols",
        "license": "upstream per-item/dataset terms; verify before redistribution",
    },
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: object) -> str:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def canonical_music_group(value: str | Path) -> str:
    """Map FMA, FakeMusicCaps and SONICS aliases to one source-song ID."""
    raw = str(value).replace("\\", "/")
    # Configured manifests may already carry the canonical value.  Preserve it
    # verbatim; running it through the filename fallbacks would otherwise turn
    # ``fmc:<ytid>`` into ``music:fmc:<ytid>`` and defeat group protection.
    if raw.startswith(("fma:", "fmc:", "sonics:", "music:")):
        return raw
    stem = Path(raw).stem
    lower = raw.lower()
    match = re.search(r"(?:sonics[_:]|fake_)(\d+)(?:_(?:suno|udio))?", lower)
    if match:
        return f"sonics:{int(match.group(1))}"
    match = re.fullmatch(r"fma[_:](\d+)", stem.lower())
    if match:
        return f"fma:{int(match.group(1))}"
    if "fma" in lower and stem.isdigit():
        return f"fma:{int(stem)}"
    if stem.isdigit() and len(stem) <= 6:
        return f"fma:{int(stem)}"
    if "fakemusiccaps" in lower:
        return f"fmc:{stem}"
    # Existing MixFake manifests retain just the FakeMusicCaps video ID.
    if re.fullmatch(r"[-_A-Za-z0-9]{11}", stem):
        return f"fmc:{stem}"
    return f"music:{stem}"


def music_identity(archive_id: str, sub_dataset: str) -> tuple[str, str, str]:
    """Return rendition ID, canonical source-song group, and generator."""
    if sub_dataset == "FMA":
        match = re.search(r"(\d{6})$", archive_id)
        if not match:
            raise ValueError(f"Malformed FMA archive ID: {archive_id}")
        source_id = match.group(1)
        return source_id, f"fma:{int(source_id)}", "FMA"
    if sub_dataset == "SONICS":
        match = re.search(r"fake_(\d+)_(suno|udio)_\d+$", archive_id.lower())
        if not match:
            raise ValueError(f"Malformed SONICS archive ID: {archive_id}")
        return archive_id, f"sonics:{int(match.group(1))}", match.group(2)
    if sub_dataset == "FakeMusicCaps":
        # YouTube IDs are exactly 11 characters and may start with '-' or '_'.
        match = re.search(r"([-_A-Za-z0-9]{11})$", archive_id)
        if not match or not archive_id.startswith("FM_"):
            raise ValueError(f"Malformed FakeMusicCaps archive ID: {archive_id}")
        source_id = match.group(1)
        generator = archive_id[3:-(len(source_id) + 1)]
        if not generator:
            raise ValueError(f"Missing FakeMusicCaps generator: {archive_id}")
        return archive_id, f"fmc:{source_id}", generator
    raise ValueError(f"Unknown music sub-dataset: {sub_dataset}")


def load_musiccaps_metadata(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"ytid", "caption", "aspect_list"}
    if missing := required - set(frame):
        raise ValueError(f"MusicCaps metadata lacks columns: {sorted(missing)}")
    if frame.ytid.duplicated().any():
        duplicates = sorted(frame.loc[frame.ytid.duplicated(False), "ytid"].unique())
        raise ValueError(f"Duplicate MusicCaps ytid values: {duplicates[:5]}")
    return {
        str(row.ytid): {
            "caption": str(row.caption), "aspect_list": str(row.aspect_list),
        }
        for row in frame.itertuples(index=False)
    }


def musiccaps_vocal_screen(
    metadata: dict[str, str] | None,
) -> tuple[bool, str]:
    """Apply the frozen caption/aspect-only instrumental rule."""
    if metadata is None:
        return False, "excluded_missing_musiccaps_metadata"
    text = " ".join(
        (metadata.get("caption", ""), metadata.get("aspect_list", ""))
    ).lower()
    explicit = any(
        re.search(pattern, text) for pattern in EXPLICIT_INSTRUMENTAL_PATTERNS
    )
    scrubbed = text
    for pattern in EXPLICIT_INSTRUMENTAL_PATTERNS:
        scrubbed = re.sub(pattern, " ", scrubbed)
    if re.search(POSITIVE_VOICE_PATTERN, scrubbed):
        return False, "excluded_positive_voice_signal"
    if not explicit:
        return False, "excluded_no_explicit_instrumental_or_no_vocal"
    return True, MUSIC_VOCAL_SCREEN_PASS


def voice_archive_id(source_id: str, authenticity: str) -> str:
    prefix = "RS" if authenticity == "bonafide" else "FS"
    return f"{prefix}_ASVspoof_{source_id}"


def parse_mixfake_protocol(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            members = [value for value in fields if "/unmixed_dataset/" in value]
            # The same protocol also enumerates already mixed files.  They are
            # intentionally irrelevant: v3 is rebuilt from the two components.
            if not members:
                continue
            labels = [value for value in fields if value in {"bonafide", "spoof"}]
            splits = [value for value in fields if value in {"train", "dev", "eval"}]
            if len(fields) < 2 or len(members) != 1 or len(labels) != 1 or len(splits) != 1:
                raise ValueError(f"Malformed MixFake protocol line {path}:{line_number}")
            if splits[0] != "eval":
                continue
            identifier = fields[1]
            member = members[0].split("/yourownpath/", 1)[-1].lstrip("/")
            row = {
                "member": member, "label": labels[0], "split": splits[0],
                "source": fields[0],
            }
            if identifier in result and result[identifier] != row:
                raise ValueError(f"Conflicting MixFake protocol ID: {identifier}")
            result[identifier] = row
    return result


def parse_asvspoof_protocol(path: Path) -> dict[str, dict[str, str]]:
    """Parse the official five-field LA protocol without assuming its filename."""
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            # Speaker IDs also start with LA_; utterance IDs use LA_E_/LA_D_/LA_T_.
            positions = [
                i for i, value in enumerate(fields)
                if re.match(r"LA_[EDT]_", value)
            ]
            label_positions = [
                i for i, value in enumerate(fields)
                if value in {"bonafide", "spoof"}
            ]
            if len(positions) != 1 or not label_positions or positions[0] == 0:
                raise ValueError(f"Malformed ASVspoof protocol line {path}:{line_number}")
            index = positions[0]
            sample_id = fields[index]
            # 2021 metadata represents bonafide as both attack and label, so
            # the rightmost authenticity token is the label.
            label_index = label_positions[-1]
            label = fields[label_index]
            between = [value for value in fields[index + 1:label_index] if value != "-"]
            attack = "bonafide" if label == "bonafide" else (between[-1] if between else "")
            if label == "spoof" and not attack:
                raise ValueError(f"Missing attack ID at {path}:{line_number}")
            row = {"speaker": fields[index - 1], "generator": attack, "label": label}
            if sample_id in result and result[sample_id] != row:
                raise ValueError(f"Conflicting ASVspoof protocol ID: {sample_id}")
            result[sample_id] = row
    if not result:
        raise ValueError(f"ASVspoof protocol is empty: {path}")
    return result


def protected_identities(config_path: Path) -> tuple[set[str], set[str]]:
    config = yaml.safe_load(config_path.read_text("utf-8")) or {}
    base = config_path.resolve().parent.parent
    exact: set[str] = set()
    music_groups: set[str] = set()
    for paths in config.values():
        if not isinstance(paths, list):
            continue
        for relative in paths:
            path = base / relative
            if not path.is_file():
                raise FileNotFoundError(f"Configured truth is missing: {path}")
            frame = pd.read_csv(path, dtype=str)
            for column in IDENTITY_COLUMNS:
                if column in frame:
                    exact.update(clean_value(value) for value in frame[column] if clean_value(value))
            for column in MUSIC_IDENTITY_COLUMNS:
                if column in frame:
                    music_groups.update(
                        canonical_music_group(value)
                        for value in frame[column] if clean_value(value)
                    )
    return exact, music_groups


def balanced_take(
    rows: list[dict[str, object]], count: int, group_key: str, seed: int,
) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[group_key])].append(row)
    for group, values in buckets.items():
        values.sort(key=lambda row: stable_rank(seed, group, row["source_id"]))
    selected: list[dict[str, object]] = []
    groups = sorted(buckets, key=lambda group: stable_rank(seed, "group", group))
    while len(selected) < count:
        progressed = False
        for group in groups:
            if buckets[group]:
                selected.append(buckets[group].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"Need {count} candidates, only found {len(selected)}")
    return selected


def balanced_unique_group_take(
    rows: list[dict[str, object]], count: int, balance_key: str,
    unique_key: str, seed: int,
) -> list[dict[str, object]]:
    """Take exact balanced quotas without reusing a canonical source group.

    FakeMusicCaps supplies several generator renditions for the same MusicCaps
    item.  Keeping only the last rendition encountered makes the apparent
    generator pool depend on protocol row order.  Select from all renditions
    instead, while retaining the holdout's canonical-group uniqueness rule.
    """
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[balance_key])].append(row)
    groups = sorted(buckets, key=lambda value: stable_rank(seed, "group", value))
    if not groups:
        raise ValueError("No candidate groups available")
    base, remainder = divmod(count, len(groups))
    targets = {
        group: base + int(index < remainder)
        for index, group in enumerate(groups)
    }
    # A canonical MusicCaps item may have renditions from several generators.
    # Greedily alternating generators can let a large bucket consume the only
    # source available to a scarce bucket even when a valid assignment exists.
    # Solve the small deterministic bipartite b-matching exactly instead.
    row_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    choices: dict[str, list[str]] = {}
    for group, values in buckets.items():
        values.sort(
            key=lambda row: stable_rank(
                seed, group, row[unique_key], row["source_id"]
            )
        )
        for row in values:
            pair = (group, str(row[unique_key]))
            row_by_pair.setdefault(pair, row)
        choices[group] = sorted(
            {str(row[unique_key]) for row in values},
            key=lambda value: stable_rank(seed, group, value),
        )
    slots = [
        (group, index)
        for group in sorted(
            groups,
            key=lambda value: (
                len(choices[value]) - targets[value],
                stable_rank(seed, "scarcity", value),
            ),
        )
        for index in range(targets[group])
    ]
    candidate_to_slot: dict[str, tuple[str, int]] = {}
    slot_to_candidate: dict[tuple[str, int], str] = {}

    def assign(slot: tuple[str, int], seen: set[str]) -> bool:
        group, _ = slot
        for candidate in choices[group]:
            if candidate in seen:
                continue
            seen.add(candidate)
            previous = candidate_to_slot.get(candidate)
            if previous is None or assign(previous, seen):
                candidate_to_slot[candidate] = slot
                slot_to_candidate[slot] = candidate
                return True
        return False

    for slot in slots:
        if not assign(slot, set()):
            group = slot[0]
            selected_for_group = sum(
                matched[0] == group for matched in slot_to_candidate
            )
            raise ValueError(
                f"Need {targets[group]} unique candidates for {group}, "
                f"only selected {selected_for_group}"
            )
    selected = [
        row_by_pair[(group, slot_to_candidate[(group, index)])]
        for group in groups
        for index in range(targets[group])
    ]
    selected_counts = Counter(str(row[balance_key]) for row in selected)
    if selected_counts != Counter(targets):
        raise AssertionError((selected_counts, targets))
    return selected


def load_semantic_exclusions(path: Path | None) -> tuple[set[str], dict[str, object]]:
    """Read an exclusion list or an acoustic-audit table without scoring it."""
    if path is None:
        return set(), {
            "applied": False, "input_rows": 0, "exclusion_rows": 0,
            "exclusion_unique_ids": 0,
        }
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "MUSIC_SOURCE_ID" not in frame:
        raise ValueError("Semantic exclusions CSV lacks MUSIC_SOURCE_ID")
    if (frame.MUSIC_SOURCE_ID.str.strip() == "").any():
        raise ValueError("Semantic exclusions CSV contains a blank MUSIC_SOURCE_ID")
    selection_rule = "all_rows"
    selected = frame
    if "ACOUSTIC_SCREEN_PASS" in frame:
        values = frame.ACOUSTIC_SCREEN_PASS.str.strip().str.lower()
        true_values = {"true", "1", "yes", "y"}
        false_values = {"false", "0", "no", "n"}
        unknown = sorted(set(values) - true_values - false_values)
        if unknown:
            raise ValueError(
                f"Unknown ACOUSTIC_SCREEN_PASS values: {unknown[:5]}"
            )
        selected = frame.loc[values.isin(false_values)]
        selection_rule = "ACOUSTIC_SCREEN_PASS=false"
    exclusions = set(selected.MUSIC_SOURCE_ID.str.strip())
    return exclusions, {
        "applied": True,
        "selection_rule": selection_rule,
        "input_rows": int(len(frame)),
        "input_unique_source_ids": int(frame.MUSIC_SOURCE_ID.nunique()),
        "exclusion_rows": int(len(selected)),
        "exclusion_unique_ids": int(len(exclusions)),
    }


def replace_selected_exclusions(
    selected: list[dict[str, object]],
    candidates: list[dict[str, object]],
    exclusions: set[str],
    group_key: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]], int]:
    """Replace only excluded selected rows, preserving every other assignment."""
    if not exclusions:
        return list(selected), [], 0
    candidate_ids = {str(row["source_id"]) for row in candidates}
    eligible_matches = len(candidate_ids & exclusions)
    selected_ids = {str(row["source_id"]) for row in selected}
    selected_groups = {
        str(row["group"]) for row in selected
        if str(row["source_id"]) not in exclusions
    }
    available: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        source_id = str(row["source_id"])
        source_group = str(row["group"])
        if (
            source_id not in exclusions
            and source_id not in selected_ids
            and source_group not in selected_groups
        ):
            available[str(row[group_key])].append(row)
    for group, rows in available.items():
        rows.sort(key=lambda row: stable_rank(seed, group, row["source_id"]))
    result = list(selected)
    replacements: list[dict[str, str]] = []
    for index, row in enumerate(result):
        source_id = str(row["source_id"])
        if source_id not in exclusions:
            continue
        group = str(row[group_key])
        if not available[group]:
            raise ValueError(f"No semantic replacement available for group {group}")
        replacement = available[group].pop(0)
        result[index] = replacement
        selected_groups.add(str(replacement["group"]))
        for candidate_group in available:
            available[candidate_group] = [
                candidate for candidate in available[candidate_group]
                if str(candidate["group"]) != str(replacement["group"])
            ]
        replacements.append({
            "excluded_source_id": source_id,
            "replacement_source_id": str(replacement["source_id"]),
            "generator": group,
        })
    if any(str(row["source_id"]) in exclusions for row in result):
        raise AssertionError("A semantic exclusion survived replacement")
    return result, replacements, eligible_matches


def component_pools(
    unmixed_details: pd.DataFrame,
    foreground_protocol: dict[str, dict[str, str]],
    background_protocol: dict[str, dict[str, str]],
    asv: dict[str, dict[str, str]] | None,
    musiccaps_metadata: dict[str, dict[str, str]],
    protected_exact: set[str],
    protected_music: set[str],
    allow_fallback: bool,
    fake_music_generators: tuple[str, ...] = FAKEMUSICCAPS_GENERATORS,
) -> tuple[dict[int, list[dict]], dict[int, list[dict]], dict[str, int]]:
    voices: dict[tuple[int, str], dict] = {}
    music: dict[tuple[int, str, str], dict] = {}
    missing_asv: set[str] = set()
    screen_counts: Counter[str] = Counter()
    for voice_aid, voice_proto in foreground_protocol.items():
        match = re.fullmatch(r"(RS|FS)_ASVspoof_(LA_E_.+)", voice_aid)
        if not match or voice_proto["source"] != "ASVspoof":
            continue
        voice_label = int(match.group(1) == "FS")
        voice_id = match.group(2)
        if (voice_proto["label"] == "spoof") != bool(voice_label):
            raise ValueError(f"Conflicting MixFake foreground protocol: {voice_aid}")
        asv_row = None if asv is None else asv.get(voice_id)
        if asv_row is None:
            missing_asv.add(voice_id)
        elif (asv_row["label"] == "spoof") != bool(voice_label):
            raise ValueError(f"ASV label conflict for {voice_id}")
        speaker = "unknown" if asv_row is None else asv_row["speaker"]
        if voice_id not in protected_exact and speaker not in protected_exact:
            voices[(voice_label, voice_id)] = {
                "source_id": voice_id,
                "archive_id": voice_aid,
                "archive_member": voice_proto["member"],
                "speaker": speaker,
                "generator": (
                    "ASVspoof2019-LA-eval-unknown"
                    if asv_row is None else asv_row["generator"]
                ),
            }

    selected_music = unmixed_details.loc[
        unmixed_details["split"].eq("eval")
        & unmixed_details.major_type.eq("Background")
        & unmixed_details.sub_type.eq("Music")
    ].copy()
    for row in selected_music.itertuples(index=False):
        screen_counts["official_eval_music_candidates"] += 1
        music_aid = Path(row.file_path).stem
        music_label = int(row.authenticity == "spoof")
        music_proto = background_protocol.get(music_aid)
        if music_proto is None or (
            music_proto["label"] == "spoof"
        ) != bool(music_label):
            raise ValueError(f"Missing/conflicting MixFake background protocol: {music_aid}")
        music_id, group, generator = music_identity(music_aid, row.sub_dataset)
        if music_label:
            screen_counts["fake_music_candidates"] += 1
            if row.sub_dataset == "SONICS":
                # SONICS Suno/Udio sources in this repository are vocal songs.
                # They cannot be called music-only under the competition label
                # definition, so none may enter this component-truth bank.
                screen_counts["excluded_sonics_suno_udio"] += 1
                continue
            if row.sub_dataset != "FakeMusicCaps":
                screen_counts["excluded_unsupported_fake_music_dataset"] += 1
                continue
            if generator not in fake_music_generators:
                screen_counts["excluded_unsupported_fake_music_generator"] += 1
                continue
            screen_counts["fake_musiccaps_candidates"] += 1
            ytid = group.removeprefix("fmc:")
            passed, vocal_screen = musiccaps_vocal_screen(
                musiccaps_metadata.get(ytid)
            )
            if not passed:
                screen_counts[vocal_screen] += 1
                continue
            screen_counts["fake_musiccaps_passed_text_screen"] += 1
        else:
            screen_counts["real_music_candidates"] += 1
            # A real vocal in the music bed cannot make VOICE_FAKE positive.
            # Retain FMA for the fake-label experiment, but expose that it was
            # not certified instrumental instead of overstating provenance.
            vocal_screen = "unscreened_real_noncausal"
        if music_id not in protected_exact and group not in protected_music:
            # Retain every fake-generator rendition until balanced selection;
            # canonical MusicCaps group uniqueness is enforced at selection.
            rendition = generator if music_label else ""
            music[(music_label, group, rendition)] = {
                "source_id": music_id,
                "group": group,
                "archive_id": music_aid,
                "archive_member": music_proto["member"],
                "generator": generator,
                "vocal_screen": vocal_screen,
            }
            screen_counts[
                "eligible_fake_after_screen_and_protection"
                if music_label else "eligible_real_after_protection"
            ] += 1
        else:
            screen_counts["excluded_protected_identity"] += 1
    if missing_asv and not allow_fallback:
        raise ValueError(
            "STRICT PROVENANCE FAILURE: official ASVspoof attack/speaker metadata "
            f"is missing for {len(missing_asv)} eval voices (examples: "
            f"{sorted(missing_asv)[:5]}). Supply --asvspoof-protocol or explicitly "
            "opt into --allow-generator-seen-fallback."
        )
    return (
        {label: [value for (kind, _), value in voices.items() if kind == label]
         for label in (0, 1)},
        {label: [value for key, value in music.items() if key[0] == label]
         for label in (0, 1)},
        dict(sorted(screen_counts.items())),
    )


def make_reservation(
    unmixed_details_path: Path,
    foreground_protocol_path: Path,
    background_protocol_path: Path,
    config_path: Path,
    asvspoof_protocol_path: Path | None,
    seed: int,
    per_cell: int = 20,
    allow_fallback: bool = False,
    musiccaps_metadata_path: Path | None = None,
    semantic_exclusions_path: Path | None = None,
    fake_music_generators: tuple[str, ...] = FAKEMUSICCAPS_GENERATORS,
    id_prefix: str = "pmv3",
) -> tuple[pd.DataFrame, dict[str, object]]:
    fake_music_generators = tuple(fake_music_generators)
    if not fake_music_generators or len(fake_music_generators) != len(
        set(fake_music_generators)
    ):
        raise ValueError("fake_music_generators must be nonempty and unique")
    unknown_generators = sorted(
        set(fake_music_generators).difference(FAKEMUSICCAPS_GENERATORS)
    )
    if unknown_generators:
        raise ValueError(
            f"Unsupported fake-music generators: {unknown_generators}"
        )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", id_prefix) is None:
        raise ValueError(f"Unsafe reservation ID prefix: {id_prefix!r}")
    if asvspoof_protocol_path is None and not allow_fallback:
        raise ValueError(
            "STRICT PROVENANCE FAILURE: --asvspoof-protocol is required. "
            "The fallback requires explicit --allow-generator-seen-fallback."
        )
    if musiccaps_metadata_path is None:
        raise ValueError(
            "STRICT SEMANTIC FAILURE: --musiccaps-metadata is required to "
            "exclude vocal fake-music sources."
        )
    input_paths = [
        unmixed_details_path, foreground_protocol_path, background_protocol_path,
        config_path, musiccaps_metadata_path,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if asvspoof_protocol_path is not None and not asvspoof_protocol_path.is_file():
        raise FileNotFoundError(asvspoof_protocol_path)
    if semantic_exclusions_path is not None and not semantic_exclusions_path.is_file():
        raise FileNotFoundError(semantic_exclusions_path)
    unmixed_details = pd.read_csv(unmixed_details_path, dtype=str)
    required = {
        "file_path", "major_type", "sub_type", "authenticity",
        "sub_dataset", "split",
    }
    if missing := required - set(unmixed_details):
        raise ValueError(f"MixFake unmixed details lacks columns: {sorted(missing)}")
    fore = parse_mixfake_protocol(foreground_protocol_path)
    back = parse_mixfake_protocol(background_protocol_path)
    asv = (
        None if asvspoof_protocol_path is None
        else parse_asvspoof_protocol(asvspoof_protocol_path)
    )
    musiccaps_metadata = load_musiccaps_metadata(musiccaps_metadata_path)
    semantic_exclusions, semantic_audit = load_semantic_exclusions(
        semantic_exclusions_path
    )
    protected_exact, protected_music = protected_identities(config_path)
    voices, music, screen_counts = component_pools(
        unmixed_details, fore, back, asv, musiccaps_metadata, protected_exact,
        protected_music, allow_fallback, fake_music_generators,
    )
    needed_per_label = len(MODES) * len(CELLS) * per_cell // 2
    # Each label occurs in two of four cells for every mode.
    if needed_per_label != 6 * per_cell:
        raise AssertionError(needed_per_label)
    selected_voice = {
        0: balanced_take(voices[0], needed_per_label, "speaker", seed + 10),
        1: balanced_take(voices[1], needed_per_label, "generator", seed + 11),
    }
    selected_music = {
        0: balanced_take(music[0], needed_per_label, "generator", seed + 20),
        1: balanced_unique_group_take(
            music[1], needed_per_label, "generator", "group", seed + 21,
        ),
    }
    selected_music[1], replacements, eligible_exclusion_matches = (
        replace_selected_exclusions(
            selected_music[1], music[1], semantic_exclusions, "generator",
            seed + 21,
        )
    )
    semantic_audit.update({
        "eligible_pool_matches": int(eligible_exclusion_matches),
        "selected_replacement_count": int(len(replacements)),
        "replacements": replacements,
    })
    offsets_voice = {0: 0, 1: 0}
    offsets_music = {0: 0, 1: 0}
    records = []
    index = 0
    for mode in MODES:
        for voice_fake, music_fake in CELLS:
            for repeat in range(per_cell):
                voice = selected_voice[voice_fake][offsets_voice[voice_fake]]
                music_item = selected_music[music_fake][offsets_music[music_fake]]
                offsets_voice[voice_fake] += 1
                offsets_music[music_fake] += 1
                key = f"{seed}|{mode}|{voice_fake}|{music_fake}|{repeat}"
                snr = (-10, -5, 0, 5, 10)[int(stable_rank(seed, key, "snr")[:8], 16) % 5]
                overlap = (0.25, 0.50, 0.75)[
                    int(stable_rank(seed, key, "overlap")[:8], 16) % 3
                ]
                order = "voice_first" if int(stable_rank(seed, key, "order")[:8], 16) % 2 == 0 else "music_first"
                gap = (0.0, 0.2, 0.5)[int(stable_rank(seed, key, "gap")[:8], 16) % 3]
                cell = f"{'F' if voice_fake else 'R'}{'F' if music_fake else 'R'}"
                records.append({
                    "BASE_ID": f"{id_prefix}_{index:04d}",
                    "FILE_FAKE": int(voice_fake or music_fake),
                    "VOICE_FAKE": voice_fake,
                    "MUSIC_FAKE": music_fake,
                    "VOICE_PRESENT": 1,
                    "MUSIC_PRESENT": 1,
                    "AUDIO_TYPE": "mixed",
                    "MIX_MODE": mode,
                    "COMPONENT_CASE": cell,
                    "EVAL_CELL": f"{mode}__{cell}",
                    "VOICE_SOURCE_ID": voice["source_id"],
                    "VOICE_SPEAKER": voice["speaker"],
                    "VOICE_GENERATOR": voice["generator"],
                    "VOICE_ARCHIVE_ID": voice["archive_id"],
                    "VOICE_ARCHIVE_MEMBER": voice["archive_member"],
                    "MUSIC_SOURCE_ID": music_item["source_id"],
                    "MUSIC_GROUP_ID": music_item["group"],
                    "MUSIC_GENERATOR": music_item["generator"],
                    "MUSIC_VOCAL_SCREEN": music_item["vocal_screen"],
                    "MUSIC_ARCHIVE_ID": music_item["archive_id"],
                    "MUSIC_ARCHIVE_MEMBER": music_item["archive_member"],
                    "SNR_DB": snr if mode != "sequential" else "",
                    "OVERLAP_FRACTION": overlap if mode == "partial_overlap" else (1.0 if mode == "concurrent" else 0.0),
                    "ORDER": order,
                    "GAP_SECONDS": gap if mode == "sequential" else 0.0,
                    "SOURCE_DATASET": "MixFake_official_eval",
                    "PROVENANCE_MODE": "fallback_generator_seen" if allow_fallback else "strict",
                })
                index += 1
    reservation = pd.DataFrame(records)
    validate_reservation(reservation, per_cell, fake_music_generators)
    paths = {
        "unmixed_details": unmixed_details_path,
        "mixfake_foreground_protocol": foreground_protocol_path,
        "mixfake_background_protocol": background_protocol_path,
        "partition_config": config_path,
        "musiccaps_metadata": musiccaps_metadata_path,
    }
    if asvspoof_protocol_path is not None:
        paths["asvspoof_protocol"] = asvspoof_protocol_path
    if semantic_exclusions_path is not None:
        paths["semantic_exclusions"] = semantic_exclusions_path
    provenance: dict[str, object] = {
        "schema_version": 1,
        "stage": "reserved",
        "seed": seed,
        "id_prefix": id_prefix,
        "per_cell": per_cell,
        "base_rows": len(reservation),
        "rendered_rows_planned": len(reservation) * len(CHANNELS),
        "channels": list(CHANNELS),
        "provenance_mode": "fallback_generator_seen" if allow_fallback else "strict",
        "generator_ood_claim": False,
        "warning": (
            "Fallback lacks official ASV attack/speaker provenance and is only "
            "source-disjoint/generator-seen."
            if allow_fallback else
            "ASV provenance is complete; fake music is limited to the explicitly "
            "declared seen FakeMusicCaps generator families passing a conservative "
            "text screen."
        ),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "licenses": list(MIXFAKE_LICENSES),
        "music_vocal_screen": {
            "version": MUSIC_VOCAL_SCREEN_VERSION,
            "pass_value": MUSIC_VOCAL_SCREEN_PASS,
            "metadata_fields": ["caption", "aspect_list"],
            "rule": (
                "Require an explicit instrumental/no-voice/no-vocal phrase and "
                "reject any remaining vocal/singing/speech/rap/choir/chant/"
                "humming/voice/lyrics/narration signal. Exclude all SONICS."
            ),
            "explicit_patterns": list(EXPLICIT_INSTRUMENTAL_PATTERNS),
            "positive_voice_pattern": POSITIVE_VOICE_PATTERN,
            "counts": screen_counts,
        },
        "semantic_exclusions": semantic_audit,
        "selection": {
            "protected_roles": "all roles in partition_config",
            "exact_identity_columns": list(IDENTITY_COLUMNS),
            "canonical_music_group_exclusion": True,
            "fake_music_generators": sorted(
                {str(row["generator"]) for row in selected_music[1]}
            ),
            "requested_fake_music_generators": list(fake_music_generators),
            "within_holdout_source_reuse": "paired channel variants only",
        },
    }
    return reservation, provenance


def validate_reservation(
    frame: pd.DataFrame,
    per_cell: int = 20,
    fake_music_generators: tuple[str, ...] = FAKEMUSICCAPS_GENERATORS,
) -> None:
    expected = len(MODES) * len(CELLS) * per_cell
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} base rows, got {len(frame)}")
    counts = frame.groupby(["MIX_MODE", "VOICE_FAKE", "MUSIC_FAKE"]).size()
    if len(counts) != 12 or set(counts) != {per_cell}:
        raise ValueError(f"Unbalanced cells: {counts.to_dict()}")
    for column in ("BASE_ID", "VOICE_SOURCE_ID", "MUSIC_GROUP_ID"):
        if frame[column].duplicated().any():
            raise ValueError(f"Source reuse in reservation column {column}")
    expected_file = frame[["VOICE_FAKE", "MUSIC_FAKE"]].max(axis=1).astype(int)
    if not np.array_equal(frame.FILE_FAKE.astype(int), expected_file):
        raise ValueError("FILE_FAKE is not the component-label OR")
    if "MUSIC_VOCAL_SCREEN" not in frame:
        raise ValueError("Reservation lacks MUSIC_VOCAL_SCREEN provenance")
    fake_music = frame.loc[frame.MUSIC_FAKE.astype(int).eq(1)]
    real_music = frame.loc[frame.MUSIC_FAKE.astype(int).eq(0)]
    if set(fake_music.MUSIC_GENERATOR) != set(fake_music_generators):
        raise ValueError(
            "Fake music is not balanced over the declared allowed generators"
        )
    generator_counts = fake_music.groupby("MUSIC_GENERATOR").size()
    if generator_counts.max() - generator_counts.min() > 1:
        raise ValueError(f"Unbalanced fake-music generators: {generator_counts.to_dict()}")
    valid_semantic_passes = {
        MUSIC_VOCAL_SCREEN_PASS,
        MUSIC_ACOUSTIC_SCREEN_PASS,
    }
    if not set(fake_music.MUSIC_VOCAL_SCREEN).issubset(valid_semantic_passes):
        raise ValueError("Fake music contains an unscreened or rejected source")
    if set(real_music.MUSIC_VOCAL_SCREEN) != {"unscreened_real_noncausal"}:
        raise ValueError("Real-music screen provenance is missing")


def write_reservation(output_dir: Path, frame: pd.DataFrame, provenance: dict) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    reservation_path = output_dir / "reservation.csv"
    frame.to_csv(reservation_path, index=False)
    provenance = dict(provenance)
    provenance["reservation_sha256"] = sha256_file(reservation_path)
    provenance["builder_sha256"] = sha256_file(Path(__file__))
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = np.nan_to_num(audio.mean(axis=1).astype(np.float32))
    if rate != SR:
        divisor = np.gcd(rate, SR)
        audio = resample_poly(audio, SR // divisor, rate // divisor).astype(np.float32)
    if not audio.size:
        raise ValueError(f"Empty source audio: {path}")
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-10))


def peak_limit(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return np.asarray(audio * min(1.0, 0.98 / max(peak, 1e-8)), dtype=np.float32)


def crop_or_tile(audio: np.ndarray, length: int, key: str) -> np.ndarray:
    if audio.size < length:
        audio = np.tile(audio, int(np.ceil(length / audio.size)))
    span = audio.size - length
    start = int(stable_rank(0, key, "crop")[:16], 16) % (span + 1)
    return np.asarray(audio[start:start + length], dtype=np.float32)


def mix_layout(voice: np.ndarray, music: np.ndarray, row: pd.Series) -> np.ndarray:
    mode = str(row.MIX_MODE)
    voice = voice[:12 * SR]
    if mode == "concurrent":
        length = max(8 * SR, len(voice))
        canvas = np.zeros(length, dtype=np.float32)
        canvas[:len(voice)] = voice
        background = crop_or_tile(music, length, f"{row.BASE_ID}|music")
        gain = rms(canvas[canvas != 0]) / (rms(background) * 10 ** (float(row.SNR_DB) / 20)) if np.any(canvas != 0) else 1.0
        return peak_limit(canvas + background * gain)
    if mode == "partial_overlap":
        music = crop_or_tile(music, 8 * SR, f"{row.BASE_ID}|music")
        overlap = max(1, int(min(len(voice), len(music)) * float(row.OVERLAP_FRACTION)))
        if row.ORDER == "voice_first":
            voice_start, music_start = 0, max(len(voice) - overlap, 0)
        else:
            music_start, voice_start = 0, max(len(music) - overlap, 0)
        length = max(voice_start + len(voice), music_start + len(music), 4 * SR)
        vc = np.zeros(length, dtype=np.float32)
        mc = np.zeros(length, dtype=np.float32)
        vc[voice_start:voice_start + len(voice)] = voice
        scale = rms(voice) / (rms(music) * 10 ** (float(row.SNR_DB) / 20))
        mc[music_start:music_start + len(music)] = music * scale
        return peak_limit(vc + mc)
    if mode == "sequential":
        music = crop_or_tile(music, 8 * SR, f"{row.BASE_ID}|music")
        gap = np.zeros(int(float(row.GAP_SECONDS) * SR), dtype=np.float32)
        parts = (voice, gap, music) if row.ORDER == "voice_first" else (music, gap, voice)
        return peak_limit(np.concatenate(parts))
    raise ValueError(f"Unknown layout: {mode}")


def locate_source(unmixed_dirs: list[Path] | Path, member: str) -> Path:
    roots = [unmixed_dirs] if isinstance(unmixed_dirs, Path) else unmixed_dirs
    relative = Path(member)
    matches: dict[Path, Path] = {}
    for root in roots:
        candidates = [
            root / relative,
            root / Path(*relative.parts[1:])
            if relative.parts and relative.parts[0] == "unmixed_dataset"
            else root / relative,
            root / relative.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                matches[candidate.resolve()] = candidate
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source for {member} across {roots}; "
            f"got {sorted(str(path) for path in matches)}"
        )
    return next(iter(matches.values()))


def build_audio(
    reservation_dir: Path,
    unmixed_dirs: list[Path] | Path,
    output_dir: Path,
    ffmpeg: Path,
    allow_fallback: bool,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    reservation_path = reservation_dir / "reservation.csv"
    provenance_path = reservation_dir / "provenance.json"
    reservation = pd.read_csv(reservation_path, dtype={"BASE_ID": str})
    provenance = json.loads(provenance_path.read_text("utf-8"))
    if sha256_file(reservation_path) != provenance["reservation_sha256"]:
        raise ValueError("Reservation hash mismatch")
    fallback = provenance["provenance_mode"] == "fallback_generator_seen"
    if fallback and not allow_fallback:
        raise ValueError(
            "Reservation uses fallback provenance; repeat with explicit "
            "--allow-generator-seen-fallback to build it."
        )
    requested_generators = tuple(
        provenance.get("selection", {}).get(
            "requested_fake_music_generators",
            provenance.get("selection", {}).get(
                "fake_music_generators", FAKEMUSICCAPS_GENERATORS
            ),
        )
    )
    validate_reservation(
        reservation, int(provenance["per_cell"]), requested_generators,
    )
    source_roots = [unmixed_dirs] if isinstance(unmixed_dirs, Path) else unmixed_dirs
    if not source_roots or any(not root.is_dir() for root in source_roots):
        raise FileNotFoundError(f"Missing source root in: {source_roots}")
    if not ffmpeg.is_file() and shutil.which(str(ffmpeg)) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    truth_rows: list[dict] = []
    source_hashes: dict[tuple[str, str], dict] = {}
    output_hashes: list[dict] = []
    for row in reservation.itertuples(index=False):
        voice_path = locate_source(source_roots, row.VOICE_ARCHIVE_MEMBER)
        music_path = locate_source(source_roots, row.MUSIC_ARCHIVE_MEMBER)
        for kind, source_id, member, path in (
            ("voice", row.VOICE_SOURCE_ID, row.VOICE_ARCHIVE_MEMBER, voice_path),
            ("music", row.MUSIC_SOURCE_ID, row.MUSIC_ARCHIVE_MEMBER, music_path),
        ):
            source_hashes.setdefault((kind, str(source_id)), {
                "KIND": kind, "SOURCE_ID": source_id, "ARCHIVE_MEMBER": member,
                "LOCAL_PATH": str(path.resolve()), "BYTES": path.stat().st_size,
                "SHA256": sha256_file(path),
            })
        mixed = mix_layout(load_audio(voice_path), load_audio(music_path), pd.Series(row._asdict()))
        if not 4 * SR <= mixed.size <= 60 * SR:
            raise ValueError(f"Out-of-range duration for {row.BASE_ID}: {mixed.size / SR}")
        for channel in CHANNELS:
            key = int(stable_rank(0, row.BASE_ID, channel)[:16], 16) % (2**32)
            rendered = apply_channel(
                mixed, channel, ffmpeg=None if channel == "clean" else ffmpeg, key=key,
            )
            sample_id = f"{row.BASE_ID}__{channel}"
            destination = audio_dir / f"{sample_id}.flac"
            sf.write(destination, rendered, SR, format="FLAC", subtype="PCM_16")
            output_hashes.append({
                "ID": sample_id, "BYTES": destination.stat().st_size,
                "SHA256": sha256_file(destination),
            })
            record = row._asdict()
            record.update({
                "ID": sample_id, "PARENT_ID": row.BASE_ID, "CHANNEL": channel,
                "DURATION": len(rendered) / SR,
            })
            truth_rows.append(record)
    truth = pd.DataFrame(truth_rows)
    expected_rendered = len(reservation) * len(CHANNELS)
    if len(truth) != expected_rendered or not (
        truth.groupby("BASE_ID").size() == len(CHANNELS)
    ).all():
        raise AssertionError(
            "Build did not produce the planned paired five-channel bases"
        )
    truth_path = output_dir / "truth.csv"
    truth.to_csv(truth_path, index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.5
    sample.to_csv(output_dir / "sample_submission.csv", index=False)
    pd.DataFrame(source_hashes.values()).sort_values(["KIND", "SOURCE_ID"]).to_csv(
        output_dir / "source_hashes.csv", index=False
    )
    pd.DataFrame(output_hashes).to_csv(output_dir / "audio_hashes.csv", index=False)
    built = dict(provenance)
    built.update({
        "stage": "built_unscored",
        "truth_sha256": sha256_file(truth_path),
        "source_hashes_sha256": sha256_file(output_dir / "source_hashes.csv"),
        "audio_hashes_sha256": sha256_file(output_dir / "audio_hashes.csv"),
        "builder_sha256_at_build": sha256_file(Path(__file__)),
        "source_roots": [str(root.resolve()) for root in source_roots],
        "ffmpeg": subprocess.run(
            [str(ffmpeg), "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        "detector_inference": False,
        "score_computed": False,
    })
    (output_dir / "provenance.json").write_text(
        json.dumps(built, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )


def default_path(relative: str) -> Path:
    return ROOT / relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reserve = sub.add_parser("reserve", help="freeze metadata-only base recipes")
    reserve.add_argument(
        "--unmixed-details", type=Path,
        default=default_path("data/external/mixfake/MixFake/protocols/unmixed_details.csv"),
    )
    reserve.add_argument(
        "--foreground-protocol", type=Path,
        default=default_path("data/external/mixfake/MixFake/protocols/Mixed_and_Fore_ForeLabel.txt"),
    )
    reserve.add_argument(
        "--background-protocol", type=Path,
        default=default_path("data/external/mixfake/MixFake/protocols/Mixed_and_Back_BackLabel.txt"),
    )
    reserve.add_argument(
        "--config", type=Path, default=default_path("configs/data_partitions.yaml")
    )
    reserve.add_argument(
        "--musiccaps-metadata", type=Path,
        default=default_path("data/sources/musiccaps_metadata/musiccaps-public.csv"),
    )
    reserve.add_argument("--asvspoof-protocol", type=Path)
    reserve.add_argument(
        "--semantic-exclusions", type=Path,
        help=(
            "CSV with MUSIC_SOURCE_ID. If ACOUSTIC_SCREEN_PASS is present, "
            "only false rows are excluded from the fake-music pool."
        ),
    )
    reserve.add_argument("--output-dir", type=Path, required=True)
    reserve.add_argument("--seed", type=int, default=20260904)
    reserve.add_argument("--per-cell", type=int, default=20)
    reserve.add_argument(
        "--id-prefix", default="pmv3",
        help="Unique sample-ID prefix for this independently registered bank.",
    )
    reserve.add_argument(
        "--fake-music-generators", nargs="+",
        choices=FAKEMUSICCAPS_GENERATORS,
        default=list(FAKEMUSICCAPS_GENERATORS),
        help=(
            "Explicit balanced FakeMusicCaps generator set. This is recorded in "
            "provenance; choose it before any detector score is inspected."
        ),
    )
    reserve.add_argument("--allow-generator-seen-fallback", action="store_true")

    build = sub.add_parser("build", help="render a frozen reservation; never score it")
    build.add_argument("--reservation-dir", type=Path, required=True)
    build.add_argument(
        "--unmixed-dir", type=Path, required=True, action="append",
        help=(
            "Extracted MixFake source root. Repeat for disjoint supplemental "
            "roots; each member must resolve in exactly one root."
        ),
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    build.add_argument("--allow-generator-seen-fallback", action="store_true")
    args = parser.parse_args()

    if args.command == "reserve":
        frame, provenance = make_reservation(
            args.unmixed_details, args.foreground_protocol, args.background_protocol,
            args.config, args.asvspoof_protocol, args.seed, args.per_cell,
            args.allow_generator_seen_fallback,
            musiccaps_metadata_path=args.musiccaps_metadata,
            semantic_exclusions_path=args.semantic_exclusions,
            fake_music_generators=tuple(args.fake_music_generators),
            id_prefix=args.id_prefix,
        )
        write_reservation(args.output_dir, frame, provenance)
        print(frame.groupby(["MIX_MODE", "COMPONENT_CASE"]).size().to_string())
        print(f"Reserved {len(frame)} unscored bases in {args.output_dir}")
    else:
        build_audio(
            args.reservation_dir, args.unmixed_dir, args.output_dir, args.ffmpeg,
            args.allow_generator_seen_fallback,
        )
        print(
            f"Built {len(pd.read_csv(args.output_dir / 'truth.csv')):,} "
            f"unscored files in {args.output_dir}"
        )


if __name__ == "__main__":
    main()

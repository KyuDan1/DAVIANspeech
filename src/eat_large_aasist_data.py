"""Leakage-aware manifests and crop targets for EAT-large–AASIST v56."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ALLOWED_ROLES = {"train", "development"}
FORBIDDEN_PATH_TOKENS = ("blind", "retrospective")


def partition_paths(config_path: Path, role: str) -> list[Path]:
    """Return only explicitly authorised train/development manifests."""
    if role not in ALLOWED_ROLES:
        raise ValueError(f"v56 may only access roles {sorted(ALLOWED_ROLES)}")
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text("utf-8")) or {}
    declared = config.get(role, [])
    if not isinstance(declared, list) or not declared:
        raise ValueError(f"partition config has no {role!r} manifests")
    root = config_path.resolve().parent.parent
    paths = [root / str(relative) for relative in declared]
    forbidden = [path for path in paths if any(
        token in str(path).lower() for token in FORBIDDEN_PATH_TOKENS
    )]
    if forbidden:
        raise ValueError(f"forbidden manifest declared in {role}: {forbidden}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def _value(row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            text = str(value).strip().lower()
            if text not in {"", "nan", "none", "0", "unknown"}:
                return text
    return "unknown"


def canonical_generator(value: str) -> str:
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "musicgen_medium": "musicgen", "musicgen_large": "musicgen",
        "audioldm2": "audioldm", "audio_ldm": "audioldm",
        "stable_audio_open": "stable_audio", "stableaudio": "stable_audio",
        "xtts_v2": "xtts", "openvoice_v2": "openvoice",
        "cosyvoice2": "cosyvoice", "f5_tts": "f5tts",
    }
    return aliases.get(value, value)


def _binary(value: object) -> int:
    return int(float(value)) if pd.notna(value) and str(value).strip() else 0


def music_generator_group(row: pd.Series) -> str:
    """One group per row, prioritising the Music generator/source identity."""
    music_present = _binary(row.get("MUSIC_PRESENT", 0))
    music_fake = _binary(row.get("MUSIC_FAKE", 0)) if music_present else 0
    if music_present and music_fake:
        generator = canonical_generator(_value(row, (
            "MUSIC_GENERATOR", "GENERATOR", "MUSIC_SOURCE_BANK",
        )))
        return f"fake_music_generator:{generator}"
    if music_present:
        identity = _value(row, (
            "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_GROUP", "GROUP_ID",
            "FMA_TRACK_ID", "SOURCE_FILE", "PARENT_ID", "ID",
        ))
        return f"real_music_identity:{identity}"
    voice_fake = _binary(row.get("VOICE_FAKE", 0))
    if voice_fake:
        generator = canonical_generator(_value(row, (
            "VOICE_GENERATOR", "FIRST_GENERATOR", "SECOND_GENERATOR", "GENERATOR",
        )))
        return f"fake_voice_generator:{generator}"
    identity = _value(row, (
        "VOICE_SOURCE_ID", "VOICE_SPEAKER", "SPEAKER", "FIRST_SOURCE_ID",
        "PARENT_ID", "GROUP_ID", "ID",
    ))
    return f"real_voice_identity:{identity}"


def _tokens_from_columns(
    row: pd.Series, namespace: str, columns: tuple[str, ...],
) -> set[str]:
    tokens = set()
    for column in columns:
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text in {"", "nan", "none", "0", "unknown", "real", "fake"}:
            continue
        tokens.add(f"{namespace}:{text}")
    return tokens


def protected_tokens(row: pd.Series) -> set[str]:
    """All voice/music generator, identity and derivative-parent views."""
    tokens = set()
    voice_present = _binary(row.get("VOICE_PRESENT", 0))
    music_present = _binary(row.get("MUSIC_PRESENT", 0))
    if voice_present:
        tokens |= _tokens_from_columns(row, "voice_identity", (
            "VOICE_SOURCE_ID", "VOICE_SPEAKER", "SPEAKER", "FIRST_SOURCE_ID",
            "SECOND_SOURCE_ID", "FIRST_GROUP", "SECOND_GROUP", "VOICE_GROUP",
        ))
        if _binary(row.get("VOICE_FAKE", 0)):
            generators = _tokens_from_columns(row, "voice_generator", (
                "VOICE_GENERATOR", "FIRST_GENERATOR", "SECOND_GENERATOR",
                "GENERATOR", "VOCODER",
            ))
            tokens |= {
                f"voice_generator:{canonical_generator(token.split(':', 1)[1])}"
                for token in generators
            }
    if music_present:
        tokens |= _tokens_from_columns(row, "music_identity", (
            "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_GROUP", "FMA_TRACK_ID",
            "ORIGINAL_AUDIO",
        ))
        if _binary(row.get("MUSIC_FAKE", 0)):
            generators = _tokens_from_columns(row, "music_generator", (
                "MUSIC_GENERATOR", "GENERATOR",
            ))
            tokens |= {
                f"music_generator:{canonical_generator(token.split(':', 1)[1])}"
                for token in generators
            }
    tokens |= _tokens_from_columns(row, "parent", (
        "PARENT_ID", "MIXTURE_ID", "PAIR_GROUP", "BASE_ID",
    ))
    # GROUP_ID is component-specific in pure speech/music banks.
    if voice_present and not music_present:
        tokens |= _tokens_from_columns(row, "voice_identity", ("GROUP_ID",))
    elif music_present and not voice_present:
        tokens |= _tokens_from_columns(row, "music_identity", ("GROUP_ID",))
    if not tokens:
        tokens.add(f"row:{row.get('ID', 'unknown')}")
    return tokens


def component_protected_tokens(row: pd.Series, component: str) -> set[str]:
    """All generator, identity and derivative-parent tokens for one task.

    The task-specific fallback must not reduce a row to one preferred column:
    derived manifests often expose the same identity under several views.  A
    component hypergraph keeps every such view atomic without connecting the
    Voice and Music identities through each other.
    """
    component = component.upper()
    if component not in {"VOICE", "MUSIC"}:
        raise ValueError("component must be VOICE or MUSIC")
    if not _binary(row.get(f"{component}_PRESENT", 0)):
        return set()
    prefix = component.lower()
    tokens = {
        token for token in protected_tokens(row)
        if token.startswith(f"{prefix}_identity:")
        or token.startswith(f"{prefix}_generator:")
        or token.startswith("parent:")
    }
    if not tokens:
        tokens.add(f"{prefix}_row:{row.get('ID', 'unknown')}")
    return tokens


def component_group(row: pd.Series, component: str) -> str | None:
    """Primary whole-generator/source group for task-specific LOGO."""
    component = component.upper()
    if component not in {"VOICE", "MUSIC"}:
        raise ValueError("component must be VOICE or MUSIC")
    if not _binary(row.get(f"{component}_PRESENT", 0)):
        return None
    if _binary(row.get(f"{component}_FAKE", 0)):
        columns = (
            ("MUSIC_GENERATOR", "GENERATOR") if component == "MUSIC" else
            ("VOICE_GENERATOR", "FIRST_GENERATOR", "SECOND_GENERATOR",
             "GENERATOR", "VOCODER")
        )
        values = sorted(_tokens_from_columns(row, "generator", columns))
        if values:
            names = "+".join(canonical_generator(value.split(":", 1)[1]) for value in values)
            return f"{component.lower()}_fake_generator:{names}"
    columns = (
        ("MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_GROUP", "FMA_TRACK_ID",
         "ORIGINAL_AUDIO", "GROUP_ID", "PARENT_ID", "ID")
        if component == "MUSIC" else
        ("VOICE_SOURCE_ID", "VOICE_SPEAKER", "SPEAKER", "FIRST_SOURCE_ID",
         "SECOND_SOURCE_ID", "FIRST_GROUP", "SECOND_GROUP", "VOICE_GROUP",
         "GROUP_ID", "PARENT_ID", "ID")
    )
    identity = _value(row, columns)
    return f"{component.lower()}_identity:{identity}"


def stable_group_fold(group: str, folds: int, seed: int) -> int:
    if folds <= 1:
        raise ValueError("fold count must exceed one")
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def group_split(
    frame: pd.DataFrame, folds: int = 3, held_out_fold: int = 0,
    seed: int = 20260904,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Generator/identity-disjoint deterministic split for Music/File training."""
    if not 0 <= held_out_fold < folds:
        raise ValueError("held-out fold is out of range")
    groups = frame.apply(music_generator_group, axis=1)
    assignment = groups.map(lambda value: stable_group_fold(value, folds, seed))
    validation = assignment.eq(held_out_fold).to_numpy()
    if not validation.any() or validation.all():
        raise ValueError("group split produced an empty partition")
    train_index = np.flatnonzero(~validation)
    validation_index = np.flatnonzero(validation)
    if set(groups.iloc[train_index]) & set(groups.iloc[validation_index]):
        raise RuntimeError("generator/identity group leaked across split")
    return train_index, validation_index, groups


@dataclass(frozen=True)
class SplitPlan:
    train_index: np.ndarray
    music_validation_index: np.ndarray
    voice_validation_index: np.ndarray
    mode: str
    audit: dict[str, int | float | str]


def _hypergraph_components(frame: pd.DataFrame) -> tuple[np.ndarray, list[set[str]]]:
    """Connected components where rows sharing any protected token are atomic."""
    size = len(frame)
    parent = np.arange(size, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owner: dict[str, int] = {}
    row_tokens = []
    for index, (_, row) in enumerate(frame.iterrows()):
        tokens = protected_tokens(row)
        row_tokens.append(tokens)
        for token in tokens:
            if token in owner:
                union(index, owner[token])
            else:
                owner[token] = index
    roots = np.asarray([find(index) for index in range(size)], dtype=np.int64)
    lookup = {root: offset for offset, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([lookup[root] for root in roots], dtype=np.int64), row_tokens


def _task_components(
    frame: pd.DataFrame, component: str,
) -> tuple[np.ndarray, list[set[str]]]:
    """Connected components over every protected identity view for one task."""
    size = len(frame)
    parent = np.arange(size, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owner: dict[str, int] = {}
    row_tokens: list[set[str]] = []
    present = np.zeros(size, dtype=bool)
    for index, (_, row) in enumerate(frame.iterrows()):
        tokens = component_protected_tokens(row, component)
        row_tokens.append(tokens)
        present[index] = bool(tokens)
        for token in tokens:
            if token in owner:
                union(index, owner[token])
            else:
                owner[token] = index
    component_ids = np.full(size, -1, dtype=np.int64)
    roots = [find(index) for index in np.flatnonzero(present)]
    lookup = {root: offset for offset, root in enumerate(dict.fromkeys(roots))}
    for index in np.flatnonzero(present):
        component_ids[index] = lookup[find(int(index))]
    return component_ids, row_tokens


def _task_validation_mask(
    frame: pd.DataFrame, component: str, folds: int, held_out_fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[set[str]]]:
    components, tokens = _task_components(frame, component)
    assignment = np.full(len(frame), -1, dtype=np.int64)
    for index in np.flatnonzero(components >= 0):
        assignment[index] = stable_group_fold(
            f"{component.lower()}_component:{components[index]}", folds, seed
        )
    return assignment == held_out_fold, components, tokens


def split_plan(
    frame: pd.DataFrame, folds: int = 3, held_out_fold: int = 0,
    seed: int = 20260904, maximum_atomic_fraction: float = 0.5,
) -> SplitPlan:
    """Use atomic hypergraph split, or task-specific LOGO if it collapses.

    Mixed FF rows can connect Voice and Music generator families into a giant
    component.  In that case a useful all-token split is impossible.  We then
    remove the union of a Music-LOGO and Voice-LOGO validation fold from train,
    and checkpoint File performance independently on both views.
    """
    components, row_tokens = _hypergraph_components(frame)
    counts = np.bincount(components)
    largest_fraction = float(counts.max() / len(frame))
    component_names = np.asarray([
        f"component:{value}" for value in components
    ])
    assignment = np.asarray([
        stable_group_fold(value, folds, seed) for value in component_names
    ])
    atomic_validation = assignment == held_out_fold
    audit: dict[str, int | float | str] = {
        "rows": len(frame), "hypergraph_components": len(counts),
        "largest_component_rows": int(counts.max()),
        "largest_component_fraction": largest_fraction,
    }

    if (
        largest_fraction <= maximum_atomic_fraction
        and atomic_validation.any() and not atomic_validation.all()
    ):
        train = np.flatnonzero(~atomic_validation)
        validation = np.flatnonzero(atomic_validation)
        train_tokens = set().union(*(row_tokens[index] for index in train))
        validation_tokens = set().union(*(row_tokens[index] for index in validation))
        overlap = train_tokens & validation_tokens
        if overlap:
            raise RuntimeError("atomic hypergraph split leaked protected tokens")
        audit.update({
            "mode": "atomic_hypergraph", "train_rows": len(train),
            "music_validation_rows": len(validation),
            "voice_validation_rows": len(validation),
            "all_token_overlap": len(overlap),
        })
        return SplitPlan(train, validation, validation, "atomic_hypergraph", audit)

    music_mask, music_components, music_tokens = _task_validation_mask(
        frame, "MUSIC", folds, held_out_fold, seed
    )
    voice_mask, voice_components, voice_tokens = _task_validation_mask(
        frame, "VOICE", folds, held_out_fold, seed + 1
    )
    train_mask = ~(music_mask | voice_mask)
    train = np.flatnonzero(train_mask)
    music_validation = np.flatnonzero(music_mask)
    voice_validation = np.flatnonzero(voice_mask)
    if not len(train) or not len(music_validation) or not len(voice_validation):
        raise ValueError("task-specific LOGO produced an empty partition")
    music_train_tokens = set().union(*(music_tokens[index] for index in train))
    music_validation_tokens = set().union(*(
        music_tokens[index] for index in music_validation
    ))
    voice_train_tokens = set().union(*(voice_tokens[index] for index in train))
    voice_validation_tokens = set().union(*(
        voice_tokens[index] for index in voice_validation
    ))
    music_overlap = music_train_tokens & music_validation_tokens
    voice_overlap = voice_train_tokens & voice_validation_tokens
    if music_overlap or voice_overlap:
        raise RuntimeError("task-specific generator/identity group leakage")
    audit.update({
        "mode": "task_specific_logo", "train_rows": len(train),
        "music_validation_rows": len(music_validation),
        "voice_validation_rows": len(voice_validation),
        "music_group_overlap": len(music_overlap),
        "voice_group_overlap": len(voice_overlap),
        "music_validation_groups": int(np.unique(
            music_components[music_validation]
        ).size),
        "voice_validation_groups": int(np.unique(
            voice_components[voice_validation]
        ).size),
        "music_components": int(np.unique(
            music_components[music_components >= 0]
        ).size),
        "voice_components": int(np.unique(
            voice_components[voice_components >= 0]
        ).size),
    })
    return SplitPlan(
        train, music_validation, voice_validation, "task_specific_logo", audit
    )


def _intervals(value: object) -> list[tuple[float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(item, (int, float)) for item in value
    ):
        value = [value]
    output = []
    if isinstance(value, (tuple, list)):
        for interval in value:
            if isinstance(interval, (tuple, list)) and len(interval) >= 2:
                start, end = float(interval[0]), float(interval[1])
                if np.isfinite(start) and np.isfinite(end) and end > start:
                    output.append((start, end))
    return output


def component_intervals(row: pd.Series, component: str) -> list[tuple[float, float]]:
    ranges = _intervals(row.get(f"{component}_RANGES"))
    if ranges:
        return ranges
    start, end = row.get(f"{component}_START"), row.get(f"{component}_END")
    if pd.notna(start) and pd.notna(end) and float(end) > float(start):
        return [(float(start), float(end))]
    if _binary(row.get(f"{component}_PRESENT", 0)):
        duration = row.get("DURATION")
        duration = float(duration) if pd.notna(duration) else float("inf")
        return [(0.0, duration)]
    return []


def _overlaps(intervals: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(max(left, start) < min(right, end) for left, right in intervals)


def crop_targets(
    row: pd.Series, crop_start_seconds: float, crop_end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Voice/Music/File fake labels and Voice/Music crop presence."""
    presence, fake = [], []
    for component in ("VOICE", "MUSIC"):
        intervals = component_intervals(row, component)
        is_present = _overlaps(intervals, crop_start_seconds, crop_end_seconds)
        presence.append(float(is_present))
        fake_ranges = _intervals(row.get(f"{component}_FAKE_RANGES"))
        if fake_ranges:
            is_fake = is_present and _overlaps(
                fake_ranges, crop_start_seconds, crop_end_seconds
            )
        else:
            label = row.get(f"{component}_FAKE")
            is_fake = is_present and pd.notna(label) and int(float(label)) == 1
        fake.append(float(is_fake))
    file_fake = float(bool(fake[0] or fake[1]))
    return (
        np.asarray([fake[0], fake[1], file_fake], dtype=np.float32),
        np.asarray(presence, dtype=np.float32),
    )

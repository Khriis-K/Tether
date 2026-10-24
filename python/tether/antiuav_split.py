"""tether-antiuav-split: a leak-free Anti-UAV300 val made of whole held-out recording sessions (ADR 0004).

The official val shares every recording session with train: its clips are the
minutes just before and after train clips of the same flight. So official train
and val are pooled, and whole sessions are held out as the new val, drawn at
random with a seed, alternating between the two recording days, until they hold
MIN_VAL_FRACTION of the pooled frames (refused if that overshoots
MAX_VAL_FRACTION). Test is kept exactly as published.

A session is a clip name minus its last two `_` fields:
`20190925_101846_1_4` -> `20190925_101846`; its first 8 characters are the day.

The split file (JSON) is what every later drone tool reads:

    {"seed": 0,
     "clips": {"train": [...], "val": [...], "test": [...]},   # "train/<clip>", "val/<clip>", "test/<clip>"
     "frames": {"<official split>/<clip>": frames, ...}}

Each clip is named by its folder under the dataset root, so a new-val clip may
live in the official train/ folder. Frame counts are recorded so the split can
be checked and re-derived from its seed without the dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_SPLIT_FILE = Path("splits/antiuav.json")
DEFAULT_SEED = 0
MIN_VAL_FRACTION = 0.15
MAX_VAL_FRACTION = 0.20

Counts = dict[str, dict[str, int]]  # official split -> clip name -> frames


def session_of(clip_name: str) -> str:
    return clip_name.rsplit("_", 2)[0]


def _clip_session(clip: str) -> str:
    """The session of a split-file clip, "<official split>/<clip name>"."""
    return session_of(clip.split("/")[1])


def make_split(counts: Counts, seed: int) -> dict:
    """Hold out whole sessions of pooled train + val as val; see the module docstring."""
    pooled = {f"{s}/{c}": n for s in ("train", "val") for c, n in counts[s].items()}
    session_frames: dict[str, int] = {}
    for clip, n in pooled.items():
        session = _clip_session(clip)
        session_frames[session] = session_frames.get(session, 0) + n

    rng = np.random.default_rng(seed)
    days = sorted({s[:8] for s in session_frames})
    by_day = [list(rng.permutation(sorted(s for s in session_frames if s[:8] == day))) for day in days]
    # Alternate days so val always has both, each day's sessions in their shuffled order.
    order = [day[i] for i in range(max(map(len, by_day), default=0)) for day in by_day if i < len(day)]

    total, val_sessions, val_frames = sum(pooled.values()), set(), 0
    for session in order:
        if val_frames >= MIN_VAL_FRACTION * total:
            break
        val_sessions.add(session)
        val_frames += session_frames[session]
    if not MIN_VAL_FRACTION * total <= val_frames <= MAX_VAL_FRACTION * total:
        raise ValueError(f"held-out sessions hold {val_frames} of {total} pooled frames, "
                         f"outside {MIN_VAL_FRACTION:.0%}-{MAX_VAL_FRACTION:.0%}")

    def is_val(clip: str) -> bool:
        return _clip_session(clip) in val_sessions

    return {"seed": seed,
            "clips": {"train": sorted(c for c in pooled if not is_val(c)),
                      "val": sorted(c for c in pooled if is_val(c)),
                      "test": sorted(f"test/{c}" for c in counts["test"])},
            "frames": {**pooled, **{f"test/{c}": n for c, n in counts["test"].items()}}}


def count_frames(root: str | Path) -> Counts:
    """Frames per clip of each official split, from the length of its `visible.json` labels."""
    root = Path(root)
    return {s: {d.name: len(json.loads((d / "visible.json").read_text(encoding="utf-8"))["exist"])
                for d in sorted((root / s).iterdir()) if d.is_dir()}
            for s in ("train", "val", "test")}


def save_split(path: str | Path, split: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(split, indent=2) + "\n")


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clip_dirs(root: str | Path, split: dict, name: str) -> list[Path]:
    """The folders of split `name`'s clips under the dataset root."""
    return [Path(root) / clip for clip in split["clips"][name]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-antiuav-split", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder (the one holding train/, val/, test/)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"default {DEFAULT_SEED}")
    parser.add_argument("--out", default=str(DEFAULT_SPLIT_FILE), help=f"default {DEFAULT_SPLIT_FILE}")
    args = parser.parse_args(argv)

    split = make_split(count_frames(args.antiuav), args.seed)
    save_split(args.out, split)
    for name, clips in split["clips"].items():
        sessions = sorted({_clip_session(c) for c in clips})
        frames = sum(split["frames"][c] for c in clips)
        print(f"{name}: {len(clips)} clips, {frames} frames, {len(sessions)} sessions")
    print(f"val sessions: {', '.join(sorted({_clip_session(c) for c in split['clips']['val']}))}")
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

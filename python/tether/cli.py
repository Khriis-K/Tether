"""tether-track: run the tracker on a detection file and write a track file."""

import argparse

from tether.formats import load_detections, save_tracks
from tether.tracker import (COAST_DAMPING, CONFIRM_FRAMES, MAX_MISSING_FRAMES, MIN_SCORE,
                            SEARCH_GATE, SEARCH_GATE_CAP, SEARCH_GATE_GROWTH, Tracker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-track", description=__doc__)
    parser.add_argument("detections", help="input detection file (.jsonl)")
    parser.add_argument("tracks", help="output track file (.jsonl)")
    parser.add_argument("--confirm-frames", type=int, default=CONFIRM_FRAMES,
                        help="matched frames in a row before a new track is confirmed "
                             f"(default {CONFIRM_FRAMES})")
    parser.add_argument("--max-missing-frames", type=int, default=MAX_MISSING_FRAMES,
                        help="most missed frames in a row a searching track survives; "
                             "it is removed on the next miss "
                             f"(default {MAX_MISSING_FRAMES})")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help=f"detections scored below this are ignored (default {MIN_SCORE})")
    parser.add_argument("--search-gate", type=float, default=SEARCH_GATE,
                        help="center-distance gate, in track box sizes, for re-matching a searching "
                             "track; this and --search-gate-growth both 0 turns it off "
                             f"(default {SEARCH_GATE})")
    parser.add_argument("--search-gate-growth", type=float, default=SEARCH_GATE_GROWTH,
                        help="added to the gate per frame the track has missed "
                             f"(default {SEARCH_GATE_GROWTH})")
    parser.add_argument("--search-gate-cap", type=float, default=SEARCH_GATE_CAP,
                        help="caps the gate at this times the distance to the nearest other "
                             f"confirmed track; inf is no cap (default {SEARCH_GATE_CAP})")
    parser.add_argument("--coast-damping", type=float, default=COAST_DAMPING,
                        help="velocity is multiplied by this each frame a track goes unmatched; "
                             f"1 is undamped, 0 holds the position still (default {COAST_DAMPING})")
    args = parser.parse_args(argv)

    tracker = Tracker(confirm_frames=args.confirm_frames,
                      max_missing_frames=args.max_missing_frames,
                      min_score=args.min_score,
                      search_gate=args.search_gate,
                      search_gate_growth=args.search_gate_growth,
                      search_gate_cap=args.search_gate_cap,
                      coast_damping=args.coast_damping)
    save_tracks(args.tracks, [tracker.update(dets) for dets in load_detections(args.detections)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

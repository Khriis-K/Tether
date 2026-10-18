"""tether-track: run the tracker on a detection file and write a track file."""

import argparse

from tether.formats import load_detections, save_tracks
from tether.tracker import CONFIRM_FRAMES, MAX_MISSING_FRAMES, Tracker


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
    args = parser.parse_args(argv)

    tracker = Tracker(confirm_frames=args.confirm_frames,
                      max_missing_frames=args.max_missing_frames)
    save_tracks(args.tracks, [tracker.update(dets) for dets in load_detections(args.detections)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

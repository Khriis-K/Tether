"""tether-track: run the tracker on a detection file and write a track file."""

import argparse

from tether.formats import load_detections, save_tracks
from tether.tracker import Tracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-track", description=__doc__)
    parser.add_argument("detections", help="input detection file (.jsonl)")
    parser.add_argument("tracks", help="output track file (.jsonl)")
    args = parser.parse_args(argv)

    tracker = Tracker()
    save_tracks(args.tracks, [tracker.update(dets) for dets in load_detections(args.detections)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

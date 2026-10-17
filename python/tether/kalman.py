"""Motion model: a constant-velocity Kalman filter over one track's box.

State is the box center and size plus their velocities: (cx, cy, w, h, vcx, vcy, vw, vh).
Noise scales with the box height, so a small far object and a large near one
are judged by the same relative standards.
"""

import numpy as np

from tether.formats import Box

POSITION_NOISE = 1 / 20  # std of measured and predicted position/size, per unit of box height
VELOCITY_NOISE = 1 / 10  # std of per-frame velocity change, per unit of box height

# A new track's velocity is unknown, so it starts with a huge spread. The first
# matched detection then sets the velocity almost entirely on its own.
_INITIAL_VELOCITY_NOISE = 10.0

_F = np.eye(8) + np.eye(8, k=4)  # each frame: position += velocity
_H = np.eye(4, 8)  # a detection measures position and size, not velocity


class BoxFilter:
    def __init__(self, box: Box) -> None:
        z = _to_measurement(box)
        self._x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        self._P = np.diag(np.square([POSITION_NOISE * h] * 4 + [_INITIAL_VELOCITY_NOISE * h] * 4))

    @property
    def box(self) -> Box:
        cx, cy, w, h = (float(v) for v in self._x[:4])
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def predict(self) -> None:
        """Step one frame forward. Uncertainty grows."""
        h = self._x[3]
        q = np.diag(np.square([POSITION_NOISE * h] * 4 + [VELOCITY_NOISE * h] * 4))
        self._x = _F @ self._x
        self._P = _F @ self._P @ _F.T + q

    def update(self, box: Box) -> None:
        """Blend in a matched detection. Uncertainty shrinks."""
        z = _to_measurement(box)
        r = np.diag(np.square([POSITION_NOISE * self._x[3]] * 4))
        s = _H @ self._P @ _H.T + r
        gain = np.linalg.solve(s, _H @ self._P).T  # P H^T S^-1, as S and P are symmetric
        self._x = self._x + gain @ (z - _H @ self._x)
        self._P = (np.eye(8) - gain @ _H) @ self._P


def _to_measurement(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

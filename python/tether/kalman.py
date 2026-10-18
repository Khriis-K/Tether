"""Motion model: a constant-velocity Kalman filter over one track's box.

State is the box center and size plus their velocities: (cx, cy, w, h, vcx, vcy, vw, vh).
Noise scales with the box height, so a small far object and a large near one
are judged by the same relative standards.
"""

import numpy as np

from tether.formats import Box

POSITION_NOISE = 1 / 20  # std of measured and predicted position/size, per unit of box height
VELOCITY_NOISE = 1 / 10  # std of per-frame change in center velocity, per unit of box height
# Real boxes change size slowly, so size velocity gets far less noise than
# center velocity. Otherwise one smaller box (a partial detection) is taken as
# a fast shrink and the next prediction is an inside-out box.
SIZE_VELOCITY_NOISE = 1 / 160

# A new track's center velocity is unknown, so it starts with a huge spread and
# the first matched detection sets it almost entirely on its own.
_INITIAL_VELOCITY_NOISE = 10.0

_F = np.eye(8) + np.eye(8, k=4)  # each frame: position += velocity
_H = np.eye(4, 8)  # a detection measures position and size, not velocity


class BoxFilter:
    def __init__(self, box: Box) -> None:
        z = _to_measurement(box)
        self._x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        self._P = _covariance(h, _INITIAL_VELOCITY_NOISE)

    @property
    def box(self) -> Box:
        cx, cy, w, h = (float(v) for v in self._x[:4])
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def predict(self) -> None:
        """Step one frame forward. Uncertainty grows."""
        q = _covariance(self._x[3], VELOCITY_NOISE)
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


def _covariance(h: float, center_velocity_noise: float) -> np.ndarray:
    """Diagonal covariance for the state, with each std scaled by box height h."""
    stds = [POSITION_NOISE] * 4 + [center_velocity_noise] * 2 + [SIZE_VELOCITY_NOISE] * 2
    return np.diag(np.square(np.array(stds) * h))


def _to_measurement(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

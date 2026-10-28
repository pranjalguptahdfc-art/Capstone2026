"""
color_model.py
--------------
Single source of truth for banana color semantics (Approach B, choose.md):
exclusive color families plus a relative brightness gate, so classification
survives exposure and color-temperature changes.

Families (a pixel is counted in at most ONE family — the hue bands do not
touch: warm ends at 34, green starts at 35):
  green : H 35-85, S ≥ 30, V ≥ 15
  warm  : H 0-34,  S ≥ 30, V ≥ 15        # one family: yellow + brown
      brown  = warm pixels with V ≤ dark    (dark relative to the crop)
      yellow = warm pixels with V >  dark    (bright relative to the crop)

  dark = BROWN_V_RATIO × V_ref,  V_ref = p95(V of the analyzed pixels),
  clamped to [30, 255]. Because the split is relative to the image's own
  brightness, a uniformly dimmed crop (LED dimmed, darker enclosure) keeps
  the same yellow/brown ratio.

  other = analyzed pixels in no family (e.g. same-colored background the
  mask let in, non-banana hues). Always ≥ 0 by construction.
"""

import numpy as np

GREEN_H = (35, 85)
WARM_H = (0, 34)         # exclusive: green starts at 35; 0-34 catches
                        # red-tinged brown spots the old brown range had
MIN_S = 30
MIN_V = 15
BROWN_V_RATIO = 0.3      # a warm pixel is "brown" when V ≤ 0.3 × V_ref
                        # measured optimum on the full 154-image held-out
                        # gate: 0.30 -> 57% agreement vs 0.40 -> 52%, with
                        # far fewer ripe/underripe mis-calls as overripe.
V_REF_FLOOR = 30.0       # clamps V_ref so a fully-dark crop can't make everything brown
V_REF_CEIL = 255.0

# Last-resort window for the MASK only (banana_mask.py): pale / whitened
# overripe skin the green+warm windows can't see.
PALE_LOWER = np.array([0, 0, 180], dtype=np.uint8)
PALE_UPPER = np.array([179, 20, 255], dtype=np.uint8)


def compute_v_ref(hsv: np.ndarray, pixel_mask: np.ndarray = None) -> float:
    """Brightness reference for a crop: p95 of V over the analyzed pixels.

    pixel_mask (uint8, 255 = analyzed) restricts the reference to the banana
    body when one is available; None uses every pixel.
    """
    v = hsv[..., 2].astype(np.float32)
    if pixel_mask is not None:
        v = v[pixel_mask > 0]
    if v.size == 0:
        return V_REF_CEIL
    ref = float(np.percentile(v, 95))
    return min(max(ref, V_REF_FLOOR), V_REF_CEIL)


def family_counts(
    hsv: np.ndarray,
    pixel_mask: np.ndarray,
    v_ref: float,
) -> dict:
    """Exclusive family counts for a crop's HSV array.

    Returns {"green", "yellow", "brown", "other", "total"} where total is
    the number of analyzed pixels (mask area when pixel_mask is given,
    else all pixels) and green + yellow + brown + other == total exactly
    (the families are disjoint and "other" is counted directly).
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = BROWN_V_RATIO * v_ref

    green = (h >= GREEN_H[0]) & (h <= GREEN_H[1]) & (s >= MIN_S) & (v >= MIN_V)
    warm = (h >= WARM_H[0]) & (h <= WARM_H[1]) & (s >= MIN_S) & (v >= MIN_V)
    brown = warm & (v <= dark)
    yellow = warm & (v > dark)
    other = ~(green | warm)          # disjoint complement

    if pixel_mask is not None:
        m = pixel_mask > 0
        green &= m
        brown &= m
        yellow &= m
        other &= m
        total = int(m.sum())
    else:
        total = int(h.size)

    return {
        "green": int(green.sum()),
        "yellow": int(yellow.sum()),
        "brown": int(brown.sum()),
        "other": int(other.sum()),
        "total": total,
    }

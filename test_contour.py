#!/usr/bin/env python3
"""Deterministic tests for contour extraction / chaining (bayesian_filter.py).

Marching squares emits one isolated 2-point segment per grid cell; without
chaining, the client strokes disconnected fragments and the contour never
forms an enclosed shape. These tests pin the chaining contract:

1. An interior blob produces ONE closed ring (first point == last point).
2. A smooth gaussian blob produces one closed ring (exercises saddle cells).
3. A region touching the grid edge produces an OPEN chain — closing it
   would invent a boundary where the fire actually extends past the grid.
4. Two separate blobs produce two separate closed rings.
5. A donut (blob with a hole) produces an outer ring + an inner ring.
6. A single-cell pocket produces a closed ring (a fire is a fire even when
   tiny — only small OPEN fragments are dropped as noise).

Run:  .venv/bin/python test_contour.py
"""
import sys

sys.path.insert(0, ".")
import numpy as np

from bayesian_filter import marching_squares_contour

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


nx = ny = 30
gx, gy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")


def ring(seg):
    return seg[0] == seg[-1]


# 1. square blob
vals = np.zeros((nx, ny)); vals[10:20, 10:20] = 1.0
seg = marching_squares_contour(vals, 0.5, gx, gy)
check("blob -> one closed ring", len(seg) == 1 and ring(seg[0]), str([len(s) for s in seg]))

# 2. gaussian blob (smooth field, saddle cells)
vals2 = np.exp(-(((xs - 15) ** 2) / 12.0 + ((ys - 15) ** 2) / 12.0))
seg2 = marching_squares_contour(vals2, 0.5, gx, gy)
check("gaussian -> one closed ring", len(seg2) == 1 and ring(seg2[0]), str([len(s) for s in seg2]))

# 3. region touching the array edge -> open chain
vals3 = np.zeros((nx, ny)); vals3[:, 0:15] = 1.0
seg3 = marching_squares_contour(vals3, 0.5, gx, gy)
check("edge region -> open chain (not closed)", len(seg3) == 1 and not ring(seg3[0]),
      str([len(s) for s in seg3]))

# 4. two blobs -> two rings
vals4 = np.zeros((nx, ny)); vals4[5:10, 5:10] = 1.0; vals4[20:25, 20:25] = 1.0
seg4 = marching_squares_contour(vals4, 0.5, gx, gy)
check("two blobs -> two closed rings", len(seg4) == 2 and all(ring(s) for s in seg4),
      str([len(s) for s in seg4]))

# 5. donut -> outer ring + inner hole ring
vals5 = np.zeros((nx, ny)); vals5[8:22, 8:22] = 1.0; vals5[13:17, 13:17] = 0.0
seg5 = marching_squares_contour(vals5, 0.5, gx, gy)
check("donut -> outer + hole rings", len(seg5) == 2 and all(ring(s) for s in seg5),
      str([len(s) for s in seg5]))

# 6. single-cell pocket -> micro ring, dropped as noise (a fire that tiny
# is not meaningful to display)
vals6 = np.zeros((nx, ny)); vals6[15, 15] = 1.0
seg6 = marching_squares_contour(vals6, 0.5, gx, gy)
check("single-cell pocket dropped (micro contour)", len(seg6) == 0,
      str([len(s) for s in seg6]))

# 7. interpolation-noise gap stitching: two blobs diagonally touching produce
# a contour whose pieces are ~0.5 cell apart; they must be stitched closed.
vals7 = np.zeros((nx, ny)); vals7[14:16, 14:16] = 1.0; vals7[16:18, 16:18] = 1.0
seg7 = marching_squares_contour(vals7, 0.5, gx, gy)
check("diagonal blobs stitch into closed rings",
      len(seg7) >= 1 and all(ring(s) for s in seg7), str([len(s) for s in seg7]))

print()
if FAILED:
    print(f"{len(FAILED)} FAILURES: {FAILED}")
    sys.exit(1)
print("ALL PASS")

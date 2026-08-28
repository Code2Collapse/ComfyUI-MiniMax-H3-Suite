"""Clip-global crop centre planner — L1 total-variation via scipy linprog.

Textbook slack-variable TV formulation; not derived from GPL MaskVidExperiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass
class TVPlannerResult:
    x: np.ndarray
    y: np.ndarray
    crop_w: float
    crop_h: float
    success: bool
    message: str


def plan_tracked_crop_tv(
    *,
    bx0: np.ndarray,
    bx1: np.ndarray,
    by0: np.ndarray,
    by1: np.ndarray,
    detected: np.ndarray,
    crop_w: float,
    crop_h: float,
    img_w: int,
    img_h: int,
    movement_cost: float = 1.0,
    center_pull: float = 1e-3,
) -> TVPlannerResult:
    """Solve for crop top-left (x_t, y_t) with fixed size and L1-TV smoothing."""
    m = int(len(bx0))
    w = float(crop_w)
    h = float(crop_h)
    if m == 0:
        return TVPlannerResult(np.zeros(0), np.zeros(0), w, h, True, "empty")

    cx = (bx0 + bx1 + 1.0) * 0.5
    cy = (by0 + by1 + 1.0) * 0.5
    pref_x = cx - w * 0.5
    pref_y = cy - h * 0.5

    off_x = 0
    off_y = m
    off_ux = 2 * m
    off_uy = off_ux + max(m - 1, 0)
    off_pxp = off_uy + max(m - 1, 0)
    off_pxm = off_pxp + m
    off_pyp = off_pxm + m
    off_pym = off_pyp + m
    nv = off_pym + m

    c = np.zeros(nv, dtype=np.float64)
    for i in range(max(m - 1, 0)):
        c[off_ux + i] = float(movement_cost)
        c[off_uy + i] = float(movement_cost)
    for i in range(m):
        if bool(detected[i]):
            c[off_pxp + i] = float(center_pull)
            c[off_pxm + i] = float(center_pull)
            c[off_pyp + i] = float(center_pull)
            c[off_pym + i] = float(center_pull)

    lo = np.full(nv, -np.inf, dtype=np.float64)
    hi = np.full(nv, np.inf, dtype=np.float64)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs_ub: list[float] = []
    rhs_eq: list[float] = []
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    row = 0

    def add_ub(coeff_cols, coeff_vals, b):
        nonlocal row
        for ci, vi in zip(coeff_cols, coeff_vals):
            rows.append(row)
            cols.append(ci)
            vals.append(vi)
        rhs_ub.append(float(b))
        row += 1

    def add_eq(coeff_cols, coeff_vals, b):
        r = len(rhs_eq)
        for ci, vi in zip(coeff_cols, coeff_vals):
            eq_rows.append(r)
            eq_cols.append(ci)
            eq_vals.append(vi)
        rhs_eq.append(float(b))

    for i in range(m - 1):
        add_ub([off_x + i + 1, off_x + i, off_ux + i], [1.0, -1.0, -1.0], 0.0)
        add_ub([off_x + i + 1, off_x + i, off_ux + i], [-1.0, 1.0, -1.0], 0.0)
        add_ub([off_ux + i], [-1.0], 0.0)

    for i in range(m - 1):
        add_ub([off_y + i + 1, off_y + i, off_uy + i], [1.0, -1.0, -1.0], 0.0)
        add_ub([off_y + i + 1, off_y + i, off_uy + i], [-1.0, 1.0, -1.0], 0.0)
        add_ub([off_uy + i], [-1.0], 0.0)

    for i in range(m):
        if not bool(detected[i]):
            continue
        add_eq(
            [off_x + i, off_pxp + i, off_pxm + i],
            [1.0, -1.0, 1.0],
            float(pref_x[i]),
        )
        add_eq(
            [off_y + i, off_pyp + i, off_pym + i],
            [1.0, -1.0, 1.0],
            float(pref_y[i]),
        )
        add_ub([off_pxp + i], [-1.0], 0.0)
        add_ub([off_pxm + i], [-1.0], 0.0)
        add_ub([off_pyp + i], [-1.0], 0.0)
        add_ub([off_pym + i], [-1.0], 0.0)

    for i in range(m):
        lo[off_x + i] = 0.0
        hi[off_x + i] = max(0.0, float(img_w) - w)
        lo[off_y + i] = 0.0
        hi[off_y + i] = max(0.0, float(img_h) - h)
        if not bool(detected[i]):
            continue
        # Containment, as two one-sided rows in linprog's `A_ub @ z <= b_ub` form.
        #   left  edge:  x <= bx0
        #   right edge:  x + w >= bx1 + 1  =>  -x <= w - bx1 - 1
        # The right-edge RHS is NEGATED relative to the natural reading. Writing
        # it as (bx1 + 1 - w) flips the inequality and makes the whole program
        # infeasible whenever the subject sits near the left of frame: at bx0=0,
        # bx1=40, w=80 it asks for x <= 0 and x >= 39 at the same time.
        # Only apply containment on an axis where the subject actually fits in
        # the crop; if it is larger, containment is unsatisfiable by definition,
        # so fall back to the centre pull and let the crop stay centred rather
        # than making the entire clip's solve fail.
        if float(bx1[i] + 1.0 - bx0[i]) <= w:
            add_ub([off_x + i], [1.0], float(bx0[i]))
            add_ub([off_x + i], [-1.0], float(w - bx1[i] - 1.0))
        if float(by1[i] + 1.0 - by0[i]) <= h:
            add_ub([off_y + i], [1.0], float(by0[i]))
            add_ub([off_y + i], [-1.0], float(h - by1[i] - 1.0))

    for i in range(max(m - 1, 0)):
        lo[off_ux + i] = 0.0
        lo[off_uy + i] = 0.0
    for i in range(m):
        lo[off_pxp + i] = 0.0
        lo[off_pxm + i] = 0.0
        lo[off_pyp + i] = 0.0
        lo[off_pym + i] = 0.0

    bounds = list(zip(lo.tolist(), hi.tolist()))
    a_ub = coo_matrix((vals, (rows, cols)), shape=(len(rhs_ub), nv)).tocsr()

    kwargs = {}
    if rhs_eq:
        a_eq = coo_matrix(
            (eq_vals, (eq_rows, eq_cols)), shape=(len(rhs_eq), nv)
        ).tocsr()
        kwargs["A_eq"] = a_eq
        kwargs["b_eq"] = np.asarray(rhs_eq, dtype=np.float64)

    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.asarray(rhs_ub, dtype=np.float64),
        bounds=bounds,
        method="highs",
        **kwargs,
    )

    if not res.success:
        x = np.clip(pref_x, 0.0, max(0.0, float(img_w) - w))
        y = np.clip(pref_y, 0.0, max(0.0, float(img_h) - h))
        return TVPlannerResult(x, y, w, h, False, res.message)

    x = res.x[off_x:off_x + m]
    y = res.x[off_y:off_y + m]
    return TVPlannerResult(x, y, w, h, True, res.message)

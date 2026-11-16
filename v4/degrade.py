"""Make a practice match's tracking look like ours.

Every target number is measured from Barca-Athletic *inputs* (the tracking itself, split into live
and dead with v2's own algorithmic output) -- never from its ground truth:

    ball missing in live play            13 %
    ball missing at stoppages            75 %   (so visible ~25 %)
    ball seen outside the pitch          in ~half of stoppages, median 3 m beyond the line
                                         (never in live play)
    length of a ball-missing gap         long bursts, median ~11 s (sampled from ours)
    players missing                      ~7 % of player-frames, in bursts

The practice tracking has the opposite habit at stoppages -- the ball is almost always missing --
so left alone a model would learn "ball missing = dead", exactly the baseline's mistake.
"""

import numpy as np

FPS = 25
LIVE_MISSING = 0.13
# Stoppages: ball visible ~25 % of the time in ours, split between "seen outside the pitch after
# going out" (in half of our stoppages, median 3 m beyond the line) and "seen at the restart spot".
OUT_SHARE = 0.5                   # share of stoppages where the ball is seen outside
OUT_BEYOND = (1.3, 3.0, 6.3)      # m beyond the line, quartiles 25 / 50 / 75 in ours
OUT_VISIBLE = 0.07                # mean share of such a stoppage it stays visible outside
RESTART_VISIBLE = 0.2             # mean share of a stoppage the ball sits at the restart spot
# Together these give, as in ours: ~25 % of stoppage frames show the ball, ~17 % of those outside.
PLAYER_MISSING = 0.07
PLAYER_GAP_S = 10.0

# Ball-missing gap lengths in our match (frames), percentiles 10..90 -- measured once from the
# tracking (see README). Sampled by interpolating between them.
GAP_PERCENTILES = np.array([12, 152, 268, 526, 845], dtype=float)
# v5 passes its own gap lengths instead: only the gaps that fall in LIVE play, measured by
# v5/measure.py. The long gaps above are mostly stoppages; using them for live play teaches the
# model that a ball missing for 30 s can still be live.


def _sample_gap(rng, gaps=GAP_PERCENTILES):
    q = rng.uniform(0.1, 0.9)
    return int(np.interp(q, [0.1, 0.25, 0.5, 0.75, 0.9], gaps))


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def degrade(match, seed, live_gaps=GAP_PERCENTILES):
    rng = np.random.default_rng(seed)
    m = dict(match)
    bx, by = match["bx"].copy(), match["by"].copy()
    labels, graded = match["labels"], match["period"] > 0
    live = graded & (labels >= 0) & (labels < 2)
    dead = graded & (labels == 2)

    length, width = match["pitch"]
    orig_bx, orig_by = match["bx"], match["by"]
    for start, stop in zip(*_runs(dead)):
        span = stop - start
        # 1a. Half of our stoppages show the ball OUTSIDE the pitch for a while after it goes
        #     out (median ~3 m beyond the line). The practice tracking drops the ball instead,
        #     so put it back: just beyond the line nearest to where it was last seen in play.
        prev = np.flatnonzero(~np.isnan(orig_bx[max(0, start - 250):start]))
        if len(prev) and rng.random() < OUT_SHARE:
            j = max(0, start - 250) + prev[-1]
            x, y = orig_bx[j], orig_by[j]
            gaps = [x, length - x, y, width - y]            # distance to each line
            side = int(np.argmin(gaps))
            beyond = float(np.interp(rng.uniform(0.25, 0.75), [0.25, 0.5, 0.75], OUT_BEYOND))
            if side == 0:
                x = -beyond
            elif side == 1:
                x = length + beyond
            elif side == 2:
                y = -beyond
            else:
                y = width + beyond
            k = int(span * rng.uniform(0, 2 * OUT_VISIBLE))
            bx[start:start + k] = x + rng.normal(0, 0.3, k)
            by[start:start + k] = y + rng.normal(0, 0.3, k)
        # 1b. Show the ball for the last part of some stoppages, sitting where the restart
        #     happens (the next visible position).
        reveal = int(span * rng.uniform(0, 2 * RESTART_VISIBLE))
        nxt = stop + np.flatnonzero(~np.isnan(orig_bx[stop:stop + 250]))[:1]
        if reveal and len(nxt):
            j = nxt[0]
            bx[stop - reveal:stop] = orig_bx[j] + rng.normal(0, 0.3)
            by[stop - reveal:stop] = orig_by[j] + rng.normal(0, 0.3)

    # 2. Live play: hide the ball in long bursts until ~13 % of live frames are missing.
    n = len(bx)
    target = LIVE_MISSING * live.sum()
    missing_live = np.isnan(bx[live]).sum()
    starts = rng.permutation(np.flatnonzero(live))
    for s in starts:
        if missing_live >= target:
            break
        if np.isnan(bx[s]):
            continue
        e = min(n, s + max(1, _sample_gap(rng, live_gaps)))
        newly = live[s:e] & ~np.isnan(bx[s:e])
        bx[s:e][newly] = np.nan
        by[s:e][newly] = np.nan
        missing_live += newly.sum()
    m["bx"], m["by"] = bx, by

    # 3. Players: drop each player in ~10 s bursts, ~7 % of the time.
    px, py = match["px"].copy(), match["py"].copy()
    gap = int(PLAYER_GAP_S * FPS)
    for k in range(px.shape[1]):
        n_bursts = int(PLAYER_MISSING * n / gap)
        for s in rng.integers(0, n - gap, n_bursts):
            px[s:s + gap, k] = np.nan
            py[s:s + gap, k] = np.nan
    m["px"], m["py"] = px, py
    return m

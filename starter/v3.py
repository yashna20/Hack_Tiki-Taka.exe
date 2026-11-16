"""v3: v2 + kick detection.

What v2 missed
--------------
v2 only asks "who is standing near the ball?". A player who intercepts the ball and plays it away
at once is near it for a split second, so v2 never sees enough evidence to hand that team the
ball, and the short possession is lost (12 found vs 17 real on the public window).

What v3 adds
------------
A *kick* (any touch that plays the ball: pass, shot, clearance, interception) shows up in the
tracking as a sudden change in the ball's velocity -- it speeds up, slows down or changes
direction. v3 finds those moments and, when a player is close enough to have caused it, gives that
player's team a strong one-off piece of evidence (W_KICK) on top of v2's proximity evidence.

So one clean kick by the other team, plus the moment of contact before it, is now enough to pay
the cost of switching possession; a player who merely runs past a moving ball (no velocity change)
still is not.

Everything else -- dead-ball evidence, switching costs, Viterbi -- is v2 unchanged.

All constants are physical (how hard a kick changes ball speed, how close the kicker must be),
or chosen so the full match has a realistic number of possessions. None are fitted to the public
5 minutes.

    python starter/v3.py                     # writes evaluation/results/v3.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import v2  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402

# ---------------------------------------------------------------- Params
FPS = 25
SMOOTH_FRAMES = 5         # moving average on ball position, to remove tracking jitter
VEL_LAG = 3               # velocity before vs after, measured this many frames apart (0.12 s)
KICK_DV = 4.0             # m/s. Change in ball velocity that counts as a kick. A gentle pass is
                          # ~5-8 m/s from standing; tracking jitter after smoothing is well below 4.
KICK_RADIUS = 2.5         # m. The kicker must be this close to the ball at the moment of the kick.
KICK_MIN_GAP = 6          # frames. Two kicks closer than this are the same kick.
KICK_SPREAD = 3           # frames either side of the kick that carry the evidence
W_KICK = 1.0              # per-frame evidence on those 7 frames -> 7 cost units per kick, about
                          # one SWITCH_HA: one kick + a little proximity flips possession.
FOLLOW_UP_S = 3.0         # s. A kick only counts as control if the same team plays the ball next,
                          # or nobody touches it again within this time (CHALLENGE.md: a deflection
                          # or block without control is not a change of possession).


def smooth_nan(x, k):
    """Centred moving average that stays NaN where the ball is missing."""
    valid = ~np.isnan(x)
    kernel = np.ones(k)
    total = np.convolve(np.where(valid, x, 0.0), kernel, "same")
    count = np.convolve(valid.astype(float), kernel, "same")
    out = np.where(count == k, total / np.maximum(count, 1), np.nan)  # only fully-seen windows
    return out


def detect_kicks(players, ball, team_of_slot, pitch):
    """Frames where the ball's velocity jumps next to a player. Returns (frames, team, dv)."""
    length, width = pitch
    bx = smooth_nan(ball[:, 0].astype(float) * length, SMOOTH_FRAMES)
    by = smooth_nan(ball[:, 1].astype(float) * width, SMOOTH_FRAMES)

    # velocity (m/s) at each frame from a centred one-frame difference
    vx = np.full_like(bx, np.nan)
    vy = np.full_like(by, np.nan)
    vx[1:-1] = (bx[2:] - bx[:-2]) * FPS / 2
    vy[1:-1] = (by[2:] - by[:-2]) * FPS / 2

    # change of velocity across the frame: after minus before
    dv = np.full_like(bx, np.nan)
    dv[VEL_LAG:-VEL_LAG] = np.hypot(vx[2 * VEL_LAG:] - vx[:-2 * VEL_LAG],
                                    vy[2 * VEL_LAG:] - vy[:-2 * VEL_LAG])
    dv0 = np.nan_to_num(dv, nan=0.0)

    # local peaks above the threshold, at least KICK_MIN_GAP apart
    candidates = np.flatnonzero(dv0 >= KICK_DV)
    peaks = []
    for f in candidates:
        lo, hi = max(0, f - KICK_MIN_GAP), f + KICK_MIN_GAP + 1
        if dv0[f] == dv0[lo:hi].max():
            if not peaks or f - peaks[-1] >= KICK_MIN_GAP:
                peaks.append(f)
    peaks = np.array(peaks, dtype=int)

    # who was close enough to have kicked it
    px = players[peaks, :, 0] * length
    py = players[peaks, :, 1] * width
    dist = np.hypot(px - bx[peaks, None], py - by[peaks, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist.argmin(axis=1)
    close = dist[np.arange(len(peaks)), nearest] <= KICK_RADIUS
    return peaks[close], team_of_slot[nearest[close]], dv0[peaks[close]]


def controlled(frames, teams):
    """Keep only kicks that look like control: the same team plays the ball next, or nobody does
    for FOLLOW_UP_S. A block or deflection is usually followed straight away by the other team."""
    keep = np.ones(len(frames), dtype=bool)
    for i in range(len(frames) - 1):
        soon = frames[i + 1] - frames[i] <= FOLLOW_UP_S * FPS
        keep[i] = not soon or teams[i + 1] == teams[i]
    return frames[keep], teams[keep]


def frame_costs(players, ball, team_of_slot, pitch):
    """v2's costs plus kick evidence."""
    cost, stats = v2.frame_costs(players, ball, team_of_slot, pitch)
    frames, teams, _ = detect_kicks(players, ball, team_of_slot, pitch)
    frames, teams = controlled(frames, teams)
    n = len(cost)
    for f, team in zip(frames, teams):
        lo, hi = max(0, f - KICK_SPREAD), min(n, f + KICK_SPREAD + 1)
        other = v2.A if team == 0 else v2.H
        cost[lo:hi, other] += W_KICK       # the kicking team's state gets relatively cheaper
        cost[lo:hi, v2.D] += W_KICK / 2    # someone kicking the ball argues it is live
    stats["kicks"] = (frames, teams)
    return cost, stats


def run(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team_of_slot = team_of_each_track(meta_path)

    print("reading tracking ...")
    players, ball = v2.read_atd(raw_path, n_frames, len(team_of_slot))
    cost, stats = frame_costs(players, ball, team_of_slot, GAMES[game]["pitch"])
    frames, teams = stats["kicks"]
    print("kicks detected: {}  (home {}, away {})".format(
        len(frames), int(np.sum(teams == 0)), int(np.sum(teams == 1))))

    labels = np.full(n_frames, "X", dtype="<U1")
    for start, end in v2.play_windows(game):
        labels[start:end + 1] = v2.LETTERS[v2.viterbi(cost[start:end + 1])]
    return labels, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v3", help="output file name (without .csv)")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    labels, _ = run()
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    v2.describe(labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
    print("\nNow:  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))
    print("      python evaluation/run_evaluation.py")

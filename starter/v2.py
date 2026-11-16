"""v2: possession as a 3-state sequence (H / A / D) decoded with Viterbi.

Why not nearest-player + majority vote (the baseline)?
  * It flips team on every brief touch, so it invents far too many possessions.
  * It calls "ball not visible" dead, but the ball is usually just hidden behind players.

What v2 does instead
--------------------
Every frame produces a *cost* for being in each state H, A, D (lower = more plausible), built from
three kinds of evidence:

  1. Control     -- a player is within CONTROL_RADIUS of the ball. Their team gets cheaper; the other
                    team and D get a little more expensive (someone playing the ball means it is live).
  2. Ball out    -- the ball is visible beyond the touchline / goal line. H and A get expensive.
  3. Tempo       -- mean player speed over a few seconds. At a stoppage players walk (~1 m/s); in live
                    play the average is ~2 m/s. Slow tempo makes D cheaper, fast tempo makes it dearer.
                    This is the only D evidence available while the ball is hidden.

Changing state costs a fixed penalty. That one idea gives all the behaviour we want:
  * a loose or hidden ball stays with the last team (no evidence -> no reason to pay a switch),
  * a single deflection/touch cannot flip possession; the other team must show sustained control,
  * the switch is placed where the new team's control *began* (Viterbi back-dates it).

All constants are set from football reasoning (walking pace, how long "control" takes), not fitted
to the public 5 minutes -- the rules forbid tuning on it and 17 possessions cannot rank variants.

    python starter/v2.py                     # writes evaluation/results/v2.csv
    python evaluation/validate_submission.py evaluation/results/v2.csv
    python evaluation/run_evaluation.py
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402

# ---------------------------------------------------------------- Params
FPS = 25
CONTROL_RADIUS = 2.5      # m. A player this close to the ball can be playing it. Generous on
                          # purpose: player and ball positions each carry ~1 m of tracking error.
OUT_MARGIN = 1.0          # m beyond the line before we believe the ball is out (tracking jitter).
TEMPO_WINDOW_S = 3.0      # s. Window for mean player speed.
WALK = 1.2                # m/s. Typical walking pace -> stoppage.
PLAY = 2.0                # m/s. Typical average player speed while the ball is live.

# Per-frame evidence weights (cost units ~ log-likelihood).
W_CONTROL = 0.35          # nearest team in control vs the other team
W_CONTROL_LIVE = 0.15     # someone touching the ball argues against D
W_OUT = 1.0               # ball visibly out of play argues against H / A
W_TEMPO = 0.12            # max per-frame push from tempo, either direction

# Switching costs. With W_CONTROL = 0.35, flipping H<->A needs roughly SWITCH_HA / 0.35 ~ 17 frames
# (~0.7 s) of net control by the other team: enough to ignore a deflection, short enough to catch an
# interception that is played away at once. Chosen so the FULL match comes out at a realistic mean
# possession of ~15 s (231 possessions), not by scoring against the public window.
SWITCH_HA = 6.0
SWITCH_LIVE_DEAD = 10.0

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def read_atd(raw_path, n_frames, n_slots):
    """players (n, slots, 3) and ball (n, 3) as x, y (normalised 0-1) and speed. NaN = missing."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # ATD counts from 1, submissions from 0
            players[i] = np.array([g.split(",") for g in parts[1].split(";")], dtype=np.float32)
            ball[i] = np.array(parts[2].split(","), dtype=np.float32)
    return players, ball


def rolling_mean(x, window):
    """Centred mean over `window` frames, ignoring NaN."""
    valid = ~np.isnan(x)
    k = np.ones(window)
    total = np.convolve(np.where(valid, x, 0.0), k, "same")
    count = np.convolve(valid.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def frame_costs(players, ball, team_of_slot, pitch):
    """(n, 3) cost of H, A, D for each frame."""
    length, width = pitch
    n = len(ball)
    px, py = players[:, :, 0] * length, players[:, :, 1] * width
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    visible = ~np.isnan(bx)

    dist = np.hypot(px - bx[:, None], py - by[:, None])          # NaN where either is missing
    home_slots, away_slots = team_of_slot == 0, team_of_slot == 1
    with np.errstate(all="ignore"):
        d_home = np.nanmin(np.where(home_slots, dist, np.inf), axis=1)
        d_away = np.nanmin(np.where(away_slots, dist, np.inf), axis=1)

    cost = np.zeros((n, 3))

    # 1. Control
    home_ctrl = visible & (d_home <= CONTROL_RADIUS) & (d_home < d_away)
    away_ctrl = visible & (d_away <= CONTROL_RADIUS) & (d_away < d_home)
    cost[home_ctrl, A] += W_CONTROL
    cost[away_ctrl, H] += W_CONTROL
    cost[home_ctrl | away_ctrl, D] += W_CONTROL_LIVE

    # 2. Ball out of play
    out = visible & ((bx < -OUT_MARGIN) | (bx > length + OUT_MARGIN) |
                     (by < -OUT_MARGIN) | (by > width + OUT_MARGIN))
    cost[out, H] += W_OUT
    cost[out, A] += W_OUT

    # 3. Tempo: mean speed of everyone tracked. Speed column is m/s in practice (values ~0-9).
    speeds = players[:, :, 2]
    n_seen = np.sum(~np.isnan(speeds), axis=1)
    with np.errstate(all="ignore"):
        mean_speed = np.where(n_seen > 0, np.nansum(speeds, axis=1) / np.maximum(n_seen, 1), np.nan)
    tempo = rolling_mean(mean_speed, int(TEMPO_WINDOW_S * FPS))
    # -1 at walking pace or slower, +1 at live-play pace or faster.
    liveness = np.clip((np.nan_to_num(tempo, nan=WALK) - WALK) / (PLAY - WALK) * 2 - 1, -1, 1)
    cost[:, D] += W_TEMPO * liveness
    cost[:, H] -= W_TEMPO * liveness / 2
    cost[:, A] -= W_TEMPO * liveness / 2

    stats = {"visible": visible, "home_ctrl": home_ctrl, "away_ctrl": away_ctrl, "out": out,
             "tempo": tempo}
    return cost, stats


def viterbi(cost):
    """Minimum-cost state path through (n, 3) frame costs with fixed switching penalties."""
    trans = np.array([[0, SWITCH_HA, SWITCH_LIVE_DEAD],
                      [SWITCH_HA, 0, SWITCH_LIVE_DEAD],
                      [SWITCH_LIVE_DEAD, SWITCH_LIVE_DEAD, 0]], dtype=float)
    n = len(cost)
    back = np.empty((n, 3), dtype=np.int8)
    score = cost[0].copy()
    for t in range(1, n):
        cand = score[:, None] + trans            # from (row) -> to (col)
        best = cand.argmin(axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + cost[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def play_windows(game):
    """Video-frame [start, end] of each half, from the tracking IdFrame span in games.py."""
    g = GAMES[game]
    return [(g["halves"][h][0] - g["offsets"][h], g["halves"][h][1] - g["offsets"][h])
            for h in (1, 2)]


def run(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team_of_slot = team_of_each_track(meta_path)

    print("reading tracking ...")
    players, ball = read_atd(raw_path, n_frames, len(team_of_slot))
    cost, stats = frame_costs(players, ball, team_of_slot, GAMES[game]["pitch"])

    labels = np.full(n_frames, "X", dtype="<U1")   # outside the halves nothing is graded
    for start, end in play_windows(game):
        path = viterbi(cost[start:end + 1])
        labels[start:end + 1] = LETTERS[path]
    return labels, stats


def describe(labels):
    graded = labels != "X"
    lab = labels[graded]
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]
    runs = lab[starts]
    poss = (runs == "H") | (runs == "A")
    durations = (ends - starts)[poss] / FPS
    print("labelled frames: H {}  A {}  D {}  ({:.0f}% dead)".format(
        *(int(np.sum(lab == x)) for x in "HAD"), 100 * np.mean(lab == "D")))
    print("possessions: {}   mean {:.1f} s   median {:.1f} s".format(
        int(poss.sum()), durations.mean(), np.median(durations)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v2", help="output file name (without .csv)")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    labels, _ = run()
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    describe(labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
    print("\nNow:  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))
    print("      python evaluation/run_evaluation.py")

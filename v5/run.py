"""v5: v4's learned "who has the ball" + v2's hand-made "is the ball dead?" hints.

    python -m v5.run experiment     # practice test, picks every setting -> v5/params.json
    python -m v5.run submit         # train on all practice matches, label Barca-Athletic once

What changes from v4
  1. Practice labels follow CHALLENGE.md's "controlled possession": the other team only takes
     the ball after a completed pass, a shot, a restart, or >= 2 s on it (v4/data.py).
  2. The per-frame cost of v4 (-log P) gets v2's rule-based evidence added, weighted by W:
       "dead"  -> only v2's dead-ball part (walking pace, ball outside the lines, nobody on
                  the ball) is added to the D column
       "full"  -> all of v2's costs (its control evidence too)
     W = 0 is plain v4. The variant, W and the two switch penalties are chosen together by
     mean spellF1 on the practice test (leave-one-match-out), never on Barca-Athletic.
  3. The practice tracking hides the ball in live play in SHORT bursts, as in our match's live
     play (measured by v5/measure.py -> v5/measured.json); v4 used all of our gaps, stoppages included.
  4. As v4: second-half frames before kick-off are D, ungraded frames are X.

Every speed is computed from positions for both the practice matches and ours, so v2's hints
mean the same thing on both.
"""

import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from v4 import data, degrade, features  # noqa: E402
from v4 import run as v4run  # noqa: E402
from v5 import measure  # noqa: E402
import v2  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

PARAMS_PATH = os.path.join(HERE, "params.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
TRAIN_SEEDS = (0, 1)
TEST_SEEDS = (10, 11)
VARIANTS = ("dead", "full")
GRID_W = (0.0, 0.5, 1.0, 2.0, 4.0)
GRID_HA = (8, 12, 16)
GRID_LD = (12, 16, 24, 32)
NOISE = 0.04            # practice gains smaller than this are treated as no change


def v2_cost(match):
    """v2's per-frame cost (n, 3) from positions only (same recipe on practice and ours)."""
    length, width = match["pitch"]
    s = features.shared(match)
    players = np.stack([match["px"] / length, match["py"] / width, s["pspeed"]], axis=2)
    ball = np.column_stack([match["bx"] / length, match["by"] / width,
                            np.full(len(match["bx"]), np.nan)])
    cost, _ = v2.frame_costs(players, ball, match["team"], match["pitch"])
    return cost


def combined_cost(prob, c2, variant, w):
    cost = -np.log(prob + v4run.EPS)
    if w == 0:
        return cost
    if variant == "full":
        return cost + w * c2
    dead_hint = c2[:, 2] - (c2[:, 0] + c2[:, 1]) / 2      # > 0: v2 thinks the ball is live
    cost = cost.copy()
    cost[:, 2] += w * dead_hint
    return cost


def decode(cost, period, s_ha, s_ld):
    out = np.full(len(cost), 3, np.int8)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if len(idx):
            a, b = idx[0], idx[-1] + 1
            out[a:b] = v4run.viterbi(cost[a:b], s_ha, s_ld)
    return out


def settings():
    yield ("v4-style", 0.0, None)
    for variant in VARIANTS:
        for w in GRID_W[1:]:
            yield (variant, w, None)


def experiment():
    t0 = time.time()
    practice = data.load_practice(controlled=True)
    live_gaps = measure.live_gap_percentiles()      # short live-play gaps, as in ours
    print("practice matches: {} (controlled-possession labels)".format(
        ", ".join(m["name"] for m in practice)))
    grid = {}         # (variant, w, ha, ld) -> list of (spell, macro, npred, ngt)
    other = {}
    for k, held in enumerate(practice):
        train = [degrade.degrade(m, seed, live_gaps) for i, m in enumerate(practice) if i != k
                 for seed in TRAIN_SEEDS]
        model = v4run.fit(train)
        for seed in TEST_SEEDS:
            test = degrade.degrade(held, seed, live_gaps)
            prob = v4run.probabilities(model, test)
            c2 = v2_cost(test)
            for variant, w, _ in settings():
                cost = combined_cost(prob, c2, variant, w)
                for ha in GRID_HA:
                    for ld in GRID_LD:
                        pred = decode(cost, test["period"], ha, ld)
                        grid.setdefault((variant, w, ha, ld), []).append(v4run.score(test, pred))
            other.setdefault("baseline_smooth50", []).append(
                v4run.score(test, v4run.baseline_smooth50(test)))
            other.setdefault("v2", []).append(v4run.score(test, v4run.run_v2_v3(test, "v2")))
        print("  held out {} done ({:.0f} s)".format(held["name"], time.time() - t0))

    def mean_spell(key):
        return np.mean([r[0] for r in grid[key]])

    best_v4 = max((k for k in grid if k[0] == "v4-style"), key=mean_spell)
    best = max(grid, key=mean_spell)

    print("\nbest switch penalties per setting (mean spellF1 over 3 held-out matches x 2 seeds)")
    print("{:<10} {:>5} {:>8} {:>8}  {}".format("variant", "W", "spellF1", "macroF1", "HA, LD"))
    for variant, w, _ in settings():
        keys = [k for k in grid if k[0] == variant and k[1] == w]
        kb = max(keys, key=mean_spell)
        r = np.array(grid[kb], float)
        print("{:<10} {:>5} {:>8.3f} {:>8.3f}  {}, {}".format(
            variant, w, r[:, 0].mean(), r[:, 1].mean(), kb[2], kb[3]))

    print("\nPRACTICE TEST (controlled labels, degraded like ours)")
    print("{:<20} {:>8} {:>8} {:>10}".format("method", "spellF1", "macroF1", "poss/true"))
    rows = [("baseline_smooth50", other["baseline_smooth50"]), ("v2", other["v2"]),
            ("v4 (new labels)", grid[best_v4]), ("v5 best", grid[best])]
    for name, res in rows:
        r = np.array(res, float)
        print("{:<20} {:>8.3f} {:>8.3f} {:>9.2f}x".format(
            name, r[:, 0].mean(), r[:, 1].mean(), (r[:, 2] / r[:, 3]).mean()))
        print("{:<20}   per run: {}".format("", " ".join("{:.2f}".format(x) for x in r[:, 0])))

    gain = mean_spell(best) - mean_spell(best_v4)
    use = best if gain > NOISE else best_v4
    print("\nv5 best {} gains {:+.3f} over v4-style {} -> {}".format(
        best, gain, best_v4,
        "USE the v2 hints" if use == best else "gain is within noise, keep W = 0"))
    params = {"variant": use[0], "w": use[1], "switch_ha": use[2], "switch_live_dead": use[3],
              "practice_spellF1": round(mean_spell(use), 4),
              "practice_spellF1_v4_style": round(mean_spell(best_v4), 4),
              "chosen_by": "mean spellF1 on the practice test (leave-one-match-out); "
                           "v2 hints kept only if they gain > {}".format(NOISE)}
    with open(PARAMS_PATH, "w") as handle:
        json.dump(params, handle, indent=2)
    print("-> {}".format(os.path.relpath(PARAMS_PATH, ROOT)))


def submit():
    from baseline import write_submission
    params = json.load(open(PARAMS_PATH))
    practice = data.load_practice(controlled=True)
    live_gaps = measure.live_gap_percentiles()
    model = v4run.fit([degrade.degrade(m, seed, live_gaps) for m in practice for seed in TRAIN_SEEDS])

    print("reading Barca-Athletic tracking ...")
    ours = data.load_ours()
    prob = v4run.probabilities(model, ours)
    cost = combined_cost(prob, v2_cost(ours), params["variant"], params["w"])
    codes = decode(cost, ours["period"], params["switch_ha"], params["switch_live_dead"])

    start2 = int(np.flatnonzero(ours["period"] == 2)[0])
    codes[start2:data.kickoff_second_half()] = 2

    labels = np.array(["H", "A", "D", "X"])[codes]
    path = write_submission(os.path.join(RESULTS_DIR, "v5.csv"), labels)
    c = codes[codes < 3]
    change = np.flatnonzero(c[1:] != c[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(c)]
    poss = c[starts] < 2
    dur = (ends - starts)[poss] / 25
    print("settings: {}".format(params))
    print("labelled: H {} A {} D {} ({:.0f}% dead)".format(
        *(int(np.sum(c == k)) for k in range(3)), 100 * np.mean(c == 2)))
    print("possessions: {}  mean {:.1f} s  median {:.1f} s".format(
        int(poss.sum()), dur.mean(), np.median(dur)))
    print("-> {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    {"experiment": experiment, "submit": submit}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()

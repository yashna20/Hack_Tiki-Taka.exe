"""v4: learned per-frame probabilities + sticky-possession decoding, chosen on practice matches.

    python -m v4.run experiment     # practice test: train on 2 matches, test on the 3rd, rotate
    python -m v4.run submit         # train on all 3, label Barca-Athletic -> evaluation/results/v4.csv

Pipeline
  1. features.py   ~70 measurements per frame from one team's point of view
  2. a gradient-boosted tree model gives P(own team / other team / dead) for every frame,
     trained ONLY on the practice matches (degraded to look like ours, degrade.py)
  3. both points of view are averaged into P(H), P(A), P(D)
  4. Viterbi: staying in a state is free, switching costs a penalty (as in v2). The two penalties
     are picked by the practice test, never by Barca-Athletic.
  5. frames of the second half before kick-off are D; ungraded frames are X.
"""

import json
import os
import sys
import time

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from v4 import data, degrade, features  # noqa: E402
import possession_metrics as pm  # noqa: E402

PARAMS_PATH = os.path.join(HERE, "params.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
TRAIN_SEEDS = (0, 1)
TEST_SEEDS = (10, 11)
ROW_STEP = 3                       # use every 3rd frame for training (neighbours are near-copies)
GRID_HA = (4, 8, 12, 16, 24)
GRID_LD = (4, 8, 12, 16, 24)
EPS = 1e-4


# ------------------------------------------------------------------ model

def training_rows(match):
    xh, xa, names = features.features(match)
    lab = match["labels"]
    rows = np.flatnonzero((match["period"] > 0) & (lab >= 0))[::ROW_STEP]
    y = lab[rows]
    y_home = np.where(y == data.D, 2, np.where(y == data.H, 0, 1))   # 0 own, 1 opp, 2 dead
    y_away = np.where(y == data.D, 2, np.where(y == data.A, 0, 1))
    return np.vstack([xh[rows], xa[rows]]), np.r_[y_home, y_away], names


def fit(matches):
    xs, ys = [], []
    for m in matches:
        x, y, names = training_rows(m)
        xs.append(x)
        ys.append(y)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)
    model.fit(np.vstack(xs), np.concatenate(ys))
    return model


def probabilities(model, match):
    xh, xa, _ = features.features(match)
    ph, pa = model.predict_proba(xh), model.predict_proba(xa)
    return np.column_stack([(ph[:, 0] + pa[:, 1]) / 2,      # H
                            (ph[:, 1] + pa[:, 0]) / 2,      # A
                            (ph[:, 2] + pa[:, 2]) / 2])     # D


# ------------------------------------------------------------------ decoding

def viterbi(cost, s_ha, s_ld):
    trans = np.array([[0, s_ha, s_ld], [s_ha, 0, s_ld], [s_ld, s_ld, 0]], float)
    n = len(cost)
    back = np.empty((n, 3), np.int8)
    score = cost[0].copy()
    cols = np.arange(3)
    for t in range(1, n):
        cand = score[:, None] + trans
        best = cand.argmin(axis=0)
        back[t] = best
        score = cand[best, cols] + cost[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(prob, period, s_ha, s_ld):
    """codes 0 H, 1 A, 2 D, 3 X (X outside the played periods)."""
    out = np.full(len(prob), 3, np.int8)
    cost = -np.log(prob + EPS)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if len(idx):
            a, b = idx[0], idx[-1] + 1
            out[a:b] = viterbi(cost[a:b], s_ha, s_ld)
    return out


# ------------------------------------------------------------------ comparisons on practice

def gt_codes(match):
    return np.where(match["labels"] >= 0, match["labels"], 3).astype(np.int8)


def score(match, pred):
    r = pm.evaluate(gt_codes(match), pred, bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def baseline_smooth50(match):
    from baseline import smooth
    s = features.shared(match)
    dist = np.where(np.isnan(s["dist"]), np.inf, s["dist"])
    nearest = match["team"][dist.argmin(axis=1)].astype(np.int8)
    codes = np.where(s["vis"], nearest, -1).astype(np.int8)
    lab = smooth(codes, 50)
    out = np.select([lab == "H", lab == "A"], [0, 1], 2).astype(np.int8)
    return np.where(match["period"] > 0, out, 3).astype(np.int8)


def run_v2_v3(match, which):
    import v2
    import v3
    length, width = match["pitch"]
    s = features.shared(match)
    players = np.stack([match["px"] / length, match["py"] / width, s["pspeed"]], axis=2)
    ball = np.column_stack([match["bx"] / length, match["by"] / width,
                            np.full(len(match["bx"]), np.nan)])
    fc = v2.frame_costs if which == "v2" else v3.frame_costs
    cost, _ = fc(players, ball, match["team"], match["pitch"])
    out = np.full(len(cost), 3, np.int8)
    for p in (1, 2):
        idx = np.flatnonzero(match["period"] == p)
        out[idx[0]:idx[-1] + 1] = v2.viterbi(cost[idx[0]:idx[-1] + 1])
    return out


def experiment():
    t0 = time.time()
    practice = data.load_practice()
    print("practice matches: {}".format(", ".join(m["name"] for m in practice)))
    results = {}          # name -> list of (spell, macro, npred, ngt)
    grid = {}             # (s_ha, s_ld) -> list of spell
    for k, held in enumerate(practice):
        train = [degrade.degrade(m, seed) for i, m in enumerate(practice) if i != k
                 for seed in TRAIN_SEEDS]
        model = fit(train)
        for seed in TEST_SEEDS:
            test = degrade.degrade(held, seed)
            prob = probabilities(model, test)
            for (a, b) in [(a, b) for a in GRID_HA for b in GRID_LD]:
                grid.setdefault((a, b), []).append(score(test, decode(prob, test["period"], a, b)))
            results.setdefault("baseline_smooth50", []).append(score(test, baseline_smooth50(test)))
            results.setdefault("v2", []).append(score(test, run_v2_v3(test, "v2")))
            results.setdefault("v3", []).append(score(test, run_v2_v3(test, "v3")))
        print("  held out {} done ({:.0f} s)".format(held["name"], time.time() - t0))

    best = max(grid, key=lambda key: np.mean([r[0] for r in grid[key]]))
    results["v4"] = grid[best]

    print("\npenalty grid, mean spellF1 over 3 held-out matches x {} seeds".format(len(TEST_SEEDS)))
    print("        " + "".join("LD={:<6}".format(b) for b in GRID_LD))
    for a in GRID_HA:
        print("HA={:<4} ".format(a) + "".join(
            "{:<9.3f}".format(np.mean([r[0] for r in grid[(a, b)]])) for b in GRID_LD))

    print("\nPRACTICE TEST (held-out matches, degraded like ours)")
    print("{:<20} {:>8} {:>8} {:>18}".format("method", "spellF1", "macroF1", "possessions/true"))
    for name in ("baseline_smooth50", "v2", "v3", "v4"):
        r = np.array(results[name], float)
        print("{:<20} {:>8.3f} {:>8.3f} {:>17.2f}x".format(
            name, r[:, 0].mean(), r[:, 1].mean(), (r[:, 2] / r[:, 3]).mean()))
        sp = r[:, 0]
        print("{:<20} {:>8}   per run: {}".format("", "", " ".join("{:.2f}".format(x) for x in sp)))

    params = {"switch_ha": best[0], "switch_live_dead": best[1],
              "chosen_by": "mean spellF1 on the practice test (leave-one-match-out)"}
    with open(PARAMS_PATH, "w") as handle:
        json.dump(params, handle, indent=2)
    print("\nchosen penalties {} -> {}".format(best, os.path.relpath(PARAMS_PATH, ROOT)))


def submit():
    from baseline import write_submission
    params = json.load(open(PARAMS_PATH))
    practice = data.load_practice()
    model = fit([degrade.degrade(m, seed) for m in practice for seed in TRAIN_SEEDS])

    print("reading Barca-Athletic tracking ...")
    ours = data.load_ours()
    prob = probabilities(model, ours)
    codes = decode(prob, ours["period"], params["switch_ha"], params["switch_live_dead"])

    # second half: graded frames before the kick-off are a stoppage
    start2 = int(np.flatnonzero(ours["period"] == 2)[0])
    codes[start2:data.kickoff_second_half()] = 2

    labels = np.array(["H", "A", "D", "X"])[codes]
    path = write_submission(os.path.join(RESULTS_DIR, "v4.csv"), labels)
    graded = codes < 3
    c = codes[graded]
    change = np.flatnonzero(c[1:] != c[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(c)]
    poss = c[starts] < 2
    dur = (ends - starts)[poss] / 25
    print("labelled: H {} A {} D {} ({:.0f}% dead)".format(
        *(int(np.sum(c == k)) for k in range(3)), 100 * np.mean(c == 2)))
    print("possessions: {}  mean {:.1f} s  median {:.1f} s".format(
        int(poss.sum()), dur.mean(), np.median(dur)))
    print("-> {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    {"experiment": experiment, "submit": submit}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()

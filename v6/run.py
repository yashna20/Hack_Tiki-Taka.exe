"""v6: v5's model and decoding, trained on 3 Metrica + 20 SkillCorner practice matches.

    sh v6/download_skillcorner.sh   # once: the SkillCorner tracking (Git LFS) -> data/external/
    python -m v6.run experiment     # practice test, picks the switch penalties -> v6/params.json
    python -m v6.run submit         # train on all 23 matches, label Barca-Athletic once

Why: v5 learnt stoppages from Metrica alone, where 27-35 % of the time is dead. In SkillCorner's
20 broadcast-tracked matches it is 37-55 %, close to what our own tracking suggests, and their
tracking loses the ball as often as ours (~40 % of frames), for real rather than simulated.

Practice test: the 23 matches are split into 4 folds (fixed, by match id). Each fold is held out
once; two models are trained on the rest and scored on it:
    v5-style   Metrica only (exactly v5's training data)
    v6         Metrica + SkillCorner
Both pick their switch penalties on the same grid. The score is the mean of the Metrica mean and
the SkillCorner mean, so neither source dominates. v6's model is used only if it beats v5-style
by more than 0.04; otherwise params.json says so and submit refuses to pretend otherwise.

Nothing here touches Barca-Athletic's ground truth, the public 5 minutes included.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from v4 import data, degrade, features  # noqa: E402
from v4 import run as v4run  # noqa: E402
from v5 import measure  # noqa: E402
from v6 import skillcorner  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

PARAMS_PATH = os.path.join(HERE, "params.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
TRAIN_SEEDS = (0, 1)          # Metrica degradations used for training (as v5)
TEST_SEED = 10                # Metrica degradation used for testing
ROW_STEP_METRICA = 3          # as v4 / v5
ROW_STEP_SC = 5               # SkillCorner is 10 fps upsampled: neighbouring rows are near-copies
N_FOLDS = 4
GRID_HA = (8, 12, 16)
GRID_LD = (8, 12, 16, 24)
NOISE = 0.04


def model():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def rows(match, step):
    xh, xa, _ = features.features(match)
    lab = match["labels"]
    idx = np.flatnonzero((match["period"] > 0) & (lab >= 0))[::step]
    y = lab[idx]
    y_home = np.where(y == data.D, 2, np.where(y == data.H, 0, 1))
    y_away = np.where(y == data.D, 2, np.where(y == data.A, 0, 1))
    return np.vstack([xh[idx], xa[idx]]), np.r_[y_home, y_away]


def fit(row_sets):
    m = model()
    m.fit(np.vstack([r[0] for r in row_sets]), np.concatenate([r[1] for r in row_sets]))
    return m


def decode(prob, period, s_ha, s_ld):
    return v5_decode(-np.log(prob + v4run.EPS), period, s_ha, s_ld)


def v5_decode(cost, period, s_ha, s_ld):
    out = np.full(len(cost), 3, np.int8)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if len(idx):
            a, b = idx[0], idx[-1] + 1
            out[a:b] = v4run.viterbi(cost[a:b], s_ha, s_ld)
    return out


def folds(names):
    """Fixed split: Metrica match k -> fold k; SkillCorner ids round-robin over the folds."""
    out = {}
    sc_i = 0
    for name in names:
        if name.startswith("practice"):
            out[name] = (int(name[-1]) - 1) % N_FOLDS
        else:
            out[name] = sc_i % N_FOLDS
            sc_i += 1
    return out


def experiment():
    t0 = time.time()
    live_gaps = measure.live_gap_percentiles()
    metrica = data.load_practice(controlled=True)
    sc_ids = skillcorner.match_ids()
    print("practice: {} Metrica + {} SkillCorner matches".format(len(metrica), len(sc_ids)))

    # training rows, computed once per match
    train_rows, tests = {}, {}
    for m in metrica:
        train_rows[m["name"]] = [rows(degrade.degrade(m, s, live_gaps), ROW_STEP_METRICA)
                                 for s in TRAIN_SEEDS]
        tests[m["name"]] = ("metrica", m)
    for mid in sc_ids:
        m = skillcorner.load(mid)
        train_rows[m["name"]] = [rows(m, ROW_STEP_SC)]
        tests[m["name"]] = ("skillcorner", m)
    print("  features ready ({:.0f} s)".format(time.time() - t0))

    fold_of = folds(list(tests))
    grid = {"v5-style": {}, "v6": {}}       # model -> (ha, ld) -> {match: score tuple}
    baseline = {}
    for k in range(N_FOLDS):
        held = [n for n in tests if fold_of[n] == k]
        rest = [n for n in tests if fold_of[n] != k]
        models = {
            "v5-style": fit([r for n in rest if n.startswith("practice") for r in train_rows[n]]),
            "v6": fit([r for n in rest for r in train_rows[n]]),
        }
        for name in held:
            source, m = tests[name]
            test = degrade.degrade(m, TEST_SEED, live_gaps) if source == "metrica" else m
            baseline[name] = v4run.score(test, v4run.baseline_smooth50(test))
            for which, mdl in models.items():
                prob = v4run.probabilities(mdl, test)
                for ha in GRID_HA:
                    for ld in GRID_LD:
                        grid[which].setdefault((ha, ld), {})[name] = v4run.score(
                            test, decode(prob, test["period"], ha, ld))
        print("  fold {} done: held out {} ({:.0f} s)".format(k, ", ".join(held), time.time() - t0))

    def by_source(scores, i=0):
        met = [v[i] for n, v in scores.items() if n.startswith("practice")]
        sc = [v[i] for n, v in scores.items() if n.startswith("sc")]
        return np.mean(met), np.mean(sc), (np.mean(met) + np.mean(sc)) / 2

    best = {w: max(grid[w], key=lambda key: by_source(grid[w][key])[2]) for w in grid}

    print("\nPRACTICE TEST (each match held out once; spellF1 / macroF1)")
    print("{:<22} {:>16} {:>16} {:>9}  {}".format(
        "method", "Metrica (3)", "SkillCorner (20)", "combined", "penalties"))
    lines = [("baseline_smooth50", baseline, "")]
    lines += [(w, grid[w][best[w]], "{}, {}".format(*best[w])) for w in ("v5-style", "v6")]
    for name, scores, pen in lines:
        s, m = by_source(scores, 0), by_source(scores, 1)
        print("{:<22} {:>7.3f} / {:.3f} {:>7.3f} / {:.3f} {:>9.3f}  {}".format(
            name, s[0], m[0], s[1], m[1], s[2], pen))

    gain = by_source(grid["v6"][best["v6"]])[2] - by_source(grid["v5-style"][best["v5-style"]])[2]
    use_v6 = gain > NOISE
    print("\nv6 gains {:+.3f} over v5-style -> {}".format(
        gain, "USE v6 (Metrica + SkillCorner)" if use_v6 else "within noise: v6 is no better than v5"))
    ha, ld = best["v6"]
    params = {"switch_ha": ha, "switch_live_dead": ld, "beats_v5_by_more_than_noise": bool(use_v6),
              "practice_combined_spellF1_v6": round(by_source(grid["v6"][best["v6"]])[2], 4),
              "practice_combined_spellF1_v5_style":
                  round(by_source(grid["v5-style"][best["v5-style"]])[2], 4),
              "chosen_by": "mean of Metrica and SkillCorner spellF1, {}-fold held-out practice "
                           "test".format(N_FOLDS)}
    with open(PARAMS_PATH, "w") as handle:
        json.dump(params, handle, indent=2)
    print("-> {}  ({:.0f} s)".format(os.path.relpath(PARAMS_PATH, ROOT), time.time() - t0))


def submit():
    from baseline import write_submission
    params = json.load(open(PARAMS_PATH))
    live_gaps = measure.live_gap_percentiles()
    sets = [rows(degrade.degrade(m, s, live_gaps), ROW_STEP_METRICA)
            for m in data.load_practice(controlled=True) for s in TRAIN_SEEDS]
    sets += [rows(skillcorner.load(mid), ROW_STEP_SC) for mid in skillcorner.match_ids()]
    mdl = fit(sets)

    print("reading Barca-Athletic tracking ...")
    ours = data.load_ours()
    prob = v4run.probabilities(mdl, ours)
    codes = decode(prob, ours["period"], params["switch_ha"], params["switch_live_dead"])
    start2 = int(np.flatnonzero(ours["period"] == 2)[0])
    codes[start2:data.kickoff_second_half()] = 2

    labels = np.array(["H", "A", "D", "X"])[codes]
    path = write_submission(os.path.join(RESULTS_DIR, "v6.csv"), labels)
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

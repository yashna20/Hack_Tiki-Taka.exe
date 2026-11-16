"""v7: v6 + controlled-possession labels on SkillCorner + a wider switch-penalty grid.

    python -m v7.run experiment     # practice test (~30 min) -> v7/params.json
    python -m v7.run submit         # train on all 23 matches, label Barca-Athletic once

Changes from v6
  1. SkillCorner labels follow CHALLENGE.md's controlled-possession rule (v7/sc_controlled.py),
     exactly like the Metrica labels; one definition across all 23 practice matches.
  2. The H<->A switch penalty grid also tries 4 and 6 (v6: 8-16), since v6 finds fewer
     possessions on our match (246) than v5 (303).

Practice test: the same 4 fixed folds as v6. Per fold two models are trained on the other folds:
    v6   Metrica + SkillCorner with phase labels
    v7   Metrica + SkillCorner with controlled labels
Held-out SkillCorner matches are scored against BOTH answer keys (phase labels and controlled
labels) so neither model is judged only by its own definition.

Nothing here touches Barca-Athletic's ground truth, the public 5 minutes included.
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

from v4 import data, degrade  # noqa: E402
from v4 import run as v4run  # noqa: E402
from v5 import measure  # noqa: E402
from v6 import skillcorner  # noqa: E402
from v6 import run as v6run  # noqa: E402
from v7 import sc_controlled  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

PARAMS_PATH = os.path.join(HERE, "params.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
GRID_HA = (4, 6, 8, 12, 16)
GRID_LD = (8, 12, 16, 24)
NOISE = 0.04


def experiment():
    t0 = time.time()
    live_gaps = measure.live_gap_percentiles()
    metrica = data.load_practice(controlled=True)
    sc_ids = skillcorner.match_ids()

    train = {"v6": {}, "v7": {}}
    tests = {}
    for m in metrica:
        r = [v6run.rows(degrade.degrade(m, s, live_gaps), v6run.ROW_STEP_METRICA)
             for s in v6run.TRAIN_SEEDS]
        train["v6"][m["name"]] = train["v7"][m["name"]] = r
        tests[m["name"]] = {"metrica": degrade.degrade(m, v6run.TEST_SEED, live_gaps)}
    for mid in sc_ids:
        phase, ctrl = skillcorner.load(mid), sc_controlled.load(mid)
        name = phase["name"]
        train["v6"][name] = [v6run.rows(phase, v6run.ROW_STEP_SC)]
        train["v7"][name] = [v6run.rows(ctrl, v6run.ROW_STEP_SC)]
        tests[name] = {"sc_phase": phase, "sc_controlled": ctrl}
    print("features ready ({:.0f} s)".format(time.time() - t0), flush=True)

    fold_of = v6run.folds(list(tests))
    # scores[model][(ha, ld)][answer_key] -> list of spellF1, macroF1
    scores = {w: {} for w in train}
    baseline = {}
    for k in range(v6run.N_FOLDS):
        held = [n for n in tests if fold_of[n] == k]
        rest = [n for n in tests if fold_of[n] != k]
        for which in train:
            mdl = v6run.fit([r for n in rest for r in train[which][n]])
            for name in held:
                keys = tests[name]
                first = next(iter(keys.values()))          # tracking is the same for both keys
                prob = v4run.probabilities(mdl, first)
                for ha in GRID_HA:
                    for ld in GRID_LD:
                        pred = v6run.decode(prob, first["period"], ha, ld)
                        for key, match in keys.items():
                            s = v4run.score(match, pred)
                            scores[which].setdefault((ha, ld), {}).setdefault(key, []).append(s[:2])
                if which == "v6":
                    pred = v4run.baseline_smooth50(first)
                    for key, match in keys.items():
                        baseline.setdefault(key, []).append(v4run.score(match, pred)[:2])
        print("fold {} done ({:.0f} s)".format(k, time.time() - t0), flush=True)

    keys = ("metrica", "sc_controlled", "sc_phase")

    def mean(res, key, i=0):
        return float(np.mean([r[i] for r in res[key]]))

    def select(res):        # Metrica and SkillCorner (controlled key) count equally
        return (mean(res, "metrica") + mean(res, "sc_controlled")) / 2

    best = {w: max(scores[w], key=lambda p: select(scores[w][p])) for w in scores}
    print("\nPRACTICE TEST (each match held out once) -- spellF1 / macroF1 per answer key")
    print("{:<20} {:>16} {:>22} {:>22}   penalties".format(
        "method", "Metrica (3)", "SkillCorner controlled", "SkillCorner phases"))
    rows = [("baseline_smooth50", baseline, "")]
    rows += [(w, scores[w][best[w]], "{}, {}".format(*best[w])) for w in ("v6", "v7")]
    # v6 with its own v6 grid, for reference
    v6_grid = {p: v for p, v in scores["v6"].items() if p[0] in (8, 12, 16)}
    b6 = max(v6_grid, key=lambda p: select(v6_grid[p]))
    rows.insert(1, ("v6 (old grid)", scores["v6"][b6], "{}, {}".format(*b6)))
    for name, res, pen in rows:
        print("{:<20} {:>7.3f} / {:.3f} {:>13.3f} / {:.3f} {:>13.3f} / {:.3f}   {}".format(
            name, *[mean(res, k, i) for k in keys for i in (0, 1)], pen))

    gain = {k: mean(scores["v7"][best["v7"]], k) - mean(scores["v6"][b6], k) for k in keys}
    print("\nv7 vs v6 spellF1: " + "  ".join("{} {:+.3f}".format(k, g) for k, g in gain.items()))
    better = gain["metrica"] > -NOISE and gain["sc_controlled"] > -NOISE and \
        max(gain["metrica"], gain["sc_controlled"]) > NOISE
    print("-> " + ("v7 is clearly better" if better else "no clear gain over v6 (within noise)"))

    ha, ld = best["v7"]
    params = {"switch_ha": ha, "switch_live_dead": ld, "clearly_better_than_v6": bool(better),
              "gain_vs_v6": {k: round(v, 4) for k, v in gain.items()},
              "chosen_by": "mean of Metrica and SkillCorner (controlled labels) spellF1, "
                           "4-fold held-out practice test"}
    with open(PARAMS_PATH, "w") as handle:
        json.dump(params, handle, indent=2)
    print("-> {}  ({:.0f} s)".format(os.path.relpath(PARAMS_PATH, ROOT), time.time() - t0))


def submit():
    from baseline import write_submission
    params = json.load(open(PARAMS_PATH))
    live_gaps = measure.live_gap_percentiles()
    sets = [v6run.rows(degrade.degrade(m, s, live_gaps), v6run.ROW_STEP_METRICA)
            for m in data.load_practice(controlled=True) for s in v6run.TRAIN_SEEDS]
    sets += [v6run.rows(sc_controlled.load(mid), v6run.ROW_STEP_SC)
             for mid in skillcorner.match_ids()]
    mdl = v6run.fit(sets)

    print("reading Barca-Athletic tracking ...")
    ours = data.load_ours()
    prob = v4run.probabilities(mdl, ours)
    codes = v6run.decode(prob, ours["period"], params["switch_ha"], params["switch_live_dead"])
    start2 = int(np.flatnonzero(ours["period"] == 2)[0])
    codes[start2:data.kickoff_second_half()] = 2

    labels = np.array(["H", "A", "D", "X"])[codes]
    path = write_submission(os.path.join(RESULTS_DIR, "v7.csv"), labels)
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

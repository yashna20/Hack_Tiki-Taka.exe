"""Measure, from the Barca-Athletic tracking alone, the numbers that make the practice matches
look like ours (v4/degrade.py), and write them to v5/measured.json.

    python -m v5.measure

Inputs only: the raw tracking, and v2's algorithmic labels (starter/v2.py) to split frames into
live and dead. No ground truth, no video, no human data about this game.

v5/run.py reads the live-play gap lengths from measured.json. The other numbers are constants in
v4/degrade.py (shared with v4, which must stay reproducible); this script recomputes them and
prints them next to the constants so anyone can check where they came from.
"""

import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from v4 import data, degrade  # noqa: E402
import v2  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

MEASURED_PATH = os.path.join(HERE, "measured.json")
PCTS = [10, 25, 50, 75, 90]


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def measure():
    ours = data.load_ours()
    graded = ours["period"] > 0
    v2_labels, _ = v2.run()                       # algorithmic live / dead split
    dead = graded & (v2_labels == "D")
    live = graded & ((v2_labels == "H") | (v2_labels == "A"))

    missing = np.isnan(ours["bx"])
    length, width = ours["pitch"]

    # ball-missing gaps, each assigned to live or dead by where most of its frames lie
    st, en = runs(missing & graded)
    lengths = en - st
    dead_share = np.array([np.mean(v2_labels[a:b] == "D") for a, b in zip(st, en)])
    live_gaps = lengths[dead_share < 0.5]

    # ball seen outside the pitch
    bx, by = ours["bx"], ours["by"]
    beyond = np.maximum.reduce([np.zeros_like(bx), -bx, bx - length, -by, by - width])
    outside = ~missing & (beyond > 0)
    ds, de = runs(dead)
    stop_out = [outside[a:b].any() for a, b in zip(ds, de)]
    visible_dead = dead & ~missing

    players = np.sum(~np.isnan(ours["px"][graded]), axis=1)

    m = {
        "ball_missing_live": float(np.mean(missing[live])),
        "ball_missing_dead": float(np.mean(missing[dead])),
        "ball_missing_graded": float(np.mean(missing[graded])),
        "gap_percentiles_all": np.percentile(lengths, PCTS).astype(int).tolist(),
        "live_gap_percentiles": np.percentile(live_gaps, PCTS).astype(int).tolist(),
        "share_of_stoppages_with_ball_outside": float(np.mean(stop_out)),
        "metres_beyond_line_q25_q50_q75":
            np.round(np.percentile(beyond[outside & dead], [25, 50, 75]), 1).tolist(),
        "share_of_visible_dead_frames_outside": float(np.mean(outside[visible_dead])),
        "players_tracked_per_frame": float(players.mean()),
        "split": "live / dead from starter/v2.py's algorithmic labels",
        "percentiles": PCTS,
    }
    with open(MEASURED_PATH, "w") as handle:
        json.dump(m, handle, indent=2)

    print("\nmeasured on Barca-Athletic tracking                 measured   used in v4/degrade.py")
    rows = [("ball missing in live play", "{:.0%}".format(m["ball_missing_live"]),
             "{:.0%}".format(degrade.LIVE_MISSING)),
            ("ball missing at stoppages", "{:.0%}".format(m["ball_missing_dead"]), "~75% (result)"),
            ("ball missing, all graded frames", "{:.0%}".format(m["ball_missing_graded"]), ""),
            ("gap length p10..p90, all gaps (frames)", str(m["gap_percentiles_all"]),
             str(degrade.GAP_PERCENTILES.astype(int).tolist()) + " (v4)"),
            ("gap length p10..p90, live play (frames)", str(m["live_gap_percentiles"]),
             "read from measured.json (v5)"),
            ("stoppages with the ball seen outside",
             "{:.0%}".format(m["share_of_stoppages_with_ball_outside"]),
             "{:.0%}".format(degrade.OUT_SHARE)),
            ("metres beyond the line q25/q50/q75", str(m["metres_beyond_line_q25_q50_q75"]),
             str(list(degrade.OUT_BEYOND))),
            ("visible stoppage frames outside",
             "{:.0%}".format(m["share_of_visible_dead_frames_outside"]), "~17% (result)"),
            ("players tracked per frame", "{:.1f}".format(m["players_tracked_per_frame"]),
             "~20.5 (result)")]
    for name, a, b in rows:
        print("  {:<44} {:>18}   {}".format(name, a, b))
    print("-> {}".format(os.path.relpath(MEASURED_PATH, ROOT)))
    return m


def live_gap_percentiles():
    if not os.path.exists(MEASURED_PATH):
        sys.exit("v5/measured.json is missing: run  python -m v5.measure  first")
    return np.array(json.load(open(MEASURED_PATH))["live_gap_percentiles"], dtype=float)


if __name__ == "__main__":
    measure()

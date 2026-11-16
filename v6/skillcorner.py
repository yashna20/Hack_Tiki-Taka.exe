"""SkillCorner open data (20 Australian A-League matches, MIT licence) as practice matches.

Download first:  sh v6/download_skillcorner.sh

Each match comes out in the same shape as v4/data.py's matches (positions in metres with the
origin in a corner, 25 frames per second), so v4's features and model work on it unchanged.

Tracking: broadcast tracking at 10 fps -> linearly interpolated to 25 fps.
    ball     only where SkillCorner *detected* it (NaN otherwise, as in ours, where the ball is
             missing on 41 % of graded frames; here ~40 %). Never interpolated across a gap
             longer than one 10 fps step.
    players  every listed player, detected on screen or extrapolated by SkillCorner (an
             algorithm). Off-screen players are an honest, algorithmic estimate, like the gaps
             and jumps in our own automatic tracking.
Labels:  SkillCorner's phases of play. A phase exists only while the ball is in play, and names
         the team in possession -> H / A inside a phase, D inside a period but outside every
         phase. These are SkillCorner's own (algorithmic) possession labels on other games.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC_DIR = os.environ.get("MTAH_SKILLCORNER_DIR") or os.path.join(
    ROOT, "data", "external", "skillcorner-opendata", "data", "matches")
CACHE_DIR = os.path.join(ROOT, "data", "external", "skillcorner-cache")
SRC_FPS, FPS = 10, 25
H, A, D, NONE = 0, 1, 2, -1


def match_ids():
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(SC_DIR, "*"))
                  if os.path.isdir(d))


def _read_source(mid):
    folder = os.path.join(SC_DIR, mid)
    meta = json.load(open(os.path.join(folder, "{}_match.json".format(mid))))
    home = meta["home_team"]["id"]
    players = [p for p in meta["players"]]
    slot = {p["id"]: i for i, p in enumerate(players)}
    team = np.array([0 if p["team_id"] == home else 1 for p in players], np.int8)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])

    path = os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(mid))
    with open(path) as handle:
        first = handle.read(200)
    if "git-lfs.github.com/spec" in first:
        raise RuntimeError("{} is a Git LFS pointer: run  sh v6/download_skillcorner.sh".format(path))

    rows = [json.loads(line) for line in open(path)]
    n = max(r["frame"] for r in rows) + 1
    px = np.full((n, len(players)), np.nan, np.float32)
    py = np.full((n, len(players)), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    for r in rows:
        f = r["frame"]
        b = r["ball_data"] or {}
        if b.get("is_detected") and b.get("x") is not None:
            bx[f], by[f] = b["x"] + length / 2, b["y"] + width / 2
        for p in r["player_data"] or []:
            k = slot.get(p["player_id"])
            if k is not None and p["x"] is not None:
                px[f, k], py[f, k] = p["x"] + length / 2, p["y"] + width / 2

    period = np.zeros(n, np.int8)
    labels = np.full(n, NONE, np.int8)
    for p in meta["match_periods"]:
        period[p["start_frame"]:p["end_frame"] + 1] = p["period"]
        labels[p["start_frame"]:p["end_frame"] + 1] = D
    phases = pd.read_csv(os.path.join(folder, "{}_phases_of_play.csv".format(mid)))
    for r in phases.itertuples(index=False):
        labels[r.frame_start:r.frame_end] = H if r.team_in_possession_id == home else A
    return {"px": px, "py": py, "bx": bx, "by": by, "team": team, "period": period,
            "labels": labels, "pitch": np.array([length, width])}


def _upsample(x, src_t, dst_t):
    """Linear interpolation along time (axis 0) that never bridges a NaN."""
    lo = np.clip(np.floor(dst_t).astype(int), 0, len(src_t) - 1)
    hi = np.clip(lo + 1, 0, len(src_t) - 1)
    w = (dst_t - lo).astype(np.float32)
    if x.ndim == 2:
        w = w[:, None]
    a, b = x[lo], x[hi]
    out = a + w * (b - a)
    # where only one neighbour exists, keep it when we are closest to it
    out = np.where(np.isnan(b) & (w < 0.5), a, out)
    out = np.where(np.isnan(a) & (w >= 0.5), b, out)
    return out.astype(np.float32)


def load(mid):
    """One SkillCorner match at 25 fps, cached in data/external/skillcorner-cache/."""
    cache = os.path.join(CACHE_DIR, "{}.npz".format(mid))
    if os.path.exists(cache):
        z = np.load(cache)
        src = {k: z[k] for k in z.files}
    else:
        src = _read_source(mid)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, **src)

    n_src = len(src["bx"])
    n = int((n_src - 1) * FPS / SRC_FPS) + 1
    t = np.arange(n) * SRC_FPS / FPS                     # position in source frames
    nearest = np.clip(np.rint(t).astype(int), 0, n_src - 1)
    return {"name": "sc{}".format(mid),
            "px": _upsample(src["px"], np.arange(n_src), t),
            "py": _upsample(src["py"], np.arange(n_src), t),
            "bx": _upsample(src["bx"], np.arange(n_src), t),
            "by": _upsample(src["by"], np.arange(n_src), t),
            "team": src["team"],
            "period": src["period"][nearest],
            "labels": src["labels"][nearest],
            "pitch": tuple(float(v) for v in src["pitch"])}


def load_all():
    return [load(mid) for mid in match_ids()]

"""SkillCorner practice matches with *controlled-possession* labels (v7).

v6 took the team in possession straight from SkillCorner's phases of play: their own algorithm,
where a touch can already hand the ball over. v7 keeps the phases only for live vs dead (no phase
= ball out of play) and decides H / A with the same rule as the Metrica labels (v4/data.py,
CHALLENGE.md's proxies):

  the other team takes the ball only once it completes a pass, shoots, restarts play, or keeps
  the ball >= 2 s before the first team plays it again. Otherwise -- a touch, a deflection, a
  won duel without control -- possession stays where it was.

The events are SkillCorner's `player_possession` rows (one per player's spell on the ball):
    ends with a successful pass              -> PASS       (confirms control)
    ends with a shot                         -> SHOT       (confirms control)
    starts from a set-piece reception        -> SET PIECE  (confirms control)
    anything else (lost, cleared, bad pass)  -> CARRY      (counts only if held >= 2 s)
Frames of a live spell before its first kept possession take the team of that possession (the
restart has a clear possessor). A live spell with no possession at all keeps the phase's team.
"""

import os

import numpy as np
import pandas as pd

from v4 import data
from v6 import skillcorner

SET_PIECE_STARTS = ("throw_in_reception", "goal_kick_reception", "free_kick_reception",
                    "corner_reception")


def _events(mid, home):
    path = os.path.join(skillcorner.SC_DIR, mid, "{}_dynamic_events.csv".format(mid))
    ev = pd.read_csv(path, low_memory=False)
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start")
    out = []
    for r in pp.itertuples(index=False):
        if r.end_type == "pass" and r.pass_outcome == "successful":
            kind = "PASS"
        elif r.end_type == "shot":
            kind = "SHOT"
        elif r.start_type in SET_PIECE_STARTS:
            kind = "SET PIECE"
        else:
            kind = "CARRY"
        # _controlled() measures time between runs in 25 fps frames: convert 10 fps -> 25 fps
        out.append({"team": 0 if r.team_id == home else 1, "type": kind,
                    "start": int(r.frame_start) * 25 // 10 + 1, "frame10": int(r.frame_start)})
    return out


def labels_10fps(mid, src):
    """Controlled labels on SkillCorner's own 10 fps frames."""
    import json
    meta = json.load(open(os.path.join(skillcorner.SC_DIR, mid, "{}_match.json".format(mid))))
    home = meta["home_team"]["id"]
    phase_lab = src["labels"]                       # v6 labels: H/A from phases, D, -1
    live = phase_lab >= 0
    live &= phase_lab < 2
    out = phase_lab.copy()

    events = _events(mid, home)
    for p in (1, 2):
        frames = np.flatnonzero(src["period"] == p)
        if not len(frames):
            continue
        evp = [e for e in events if frames[0] <= e["frame10"] <= frames[-1]]
        kept = data._controlled(evp, frames[-1] * 25 // 10 + 1)
        if not kept:
            continue
        k_frames = np.array([e["frame10"] for e in kept])
        k_teams = np.array([e["team"] for e in kept])
        # team of the latest kept possession at every frame of this period
        idx = np.searchsorted(k_frames, frames, side="right") - 1
        team_now = np.where(idx >= 0, k_teams[np.maximum(idx, 0)], -1)

        # live spells: before the spell's first kept possession, use that possession's team
        d = np.diff(np.r_[0, live[frames].astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            f0, f1 = frames[a], frames[b - 1] + 1
            inside = np.flatnonzero((k_frames >= f0) & (k_frames < f1))
            if not len(inside):
                continue                            # no possession event: keep the phase team
            seg = team_now[a:b].copy()
            first = k_frames[inside[0]]
            seg[:first - f0] = k_teams[inside[0]]
            out[f0:f1] = seg
    return out


def load(mid):
    """Like v6.skillcorner.load, but with controlled-possession labels."""
    m = skillcorner.load(mid)
    z = np.load(os.path.join(skillcorner.CACHE_DIR, "{}.npz".format(mid)))
    src = {k: z[k] for k in z.files}
    lab10 = labels_10fps(mid, src)
    n = len(m["bx"])
    nearest = np.clip(np.rint(np.arange(n) * skillcorner.SRC_FPS / skillcorner.FPS).astype(int),
                      0, len(lab10) - 1)
    m = dict(m)
    m["labels"] = lab10[nearest].astype(np.int8)
    m["name"] = m["name"] + "c"
    return m

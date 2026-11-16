"""Load matches into one common shape, for our match and the three practice matches.

A match is a dict:
    px, py   (n, slots) player positions in metres, NaN where missing
    bx, by   (n,)       ball position in metres, NaN where missing
    team     (slots,)   0 = home, 1 = away
    period   (n,)       1 / 2 for frames that are played, 0 otherwise
    labels   (n,)       0 H, 1 A, 2 D, -1 unknown   (practice matches only)
    pitch    (length, width) in metres

Practice matches: Metrica Sports' public sample data
(https://github.com/metrica-sports/sample-data) -- three anonymised matches with tracking and
human-made event data. The rules allow human labels about *other* games. Their labels here are
derived mechanically from those events (see labels_from_events), never from Barca-Athletic.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402

EXTERNAL_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    ROOT, "data", "external", "sample-data", "data")
PRACTICE_PITCH = (105.0, 68.0)

H, A, D, NONE = 0, 1, 2, -1

# Event types that mean "this team is playing the ball now".
ESTABLISH = {"PASS", "SHOT", "CARRY", "RECOVERY", "BALL LOST", "SET PIECE"}


# ------------------------------------------------------------------ our match

def kickoff_second_half():
    """Video frame of the 2nd-half kick-off, from the anchor in games.py (DATA.md / FAQ.md)."""
    k = GAMES[GAME]["kickoff_2nd"]
    return (k["minute"] * 60 + k["second"]) * GAMES[GAME]["fps"] + k["frame"] + 1


def play_windows(game=GAME):
    """Video-frame [start, end] of each half, from the tracking IdFrame span in games.py."""
    g = GAMES[game]
    return [(g["halves"][h][0] - g["offsets"][h], g["halves"][h][1] - g["offsets"][h])
            for h in (1, 2)]


def load_ours(game=GAME, data_dir=None):
    from baseline import team_of_each_track
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]

    px = np.full((n, len(team)), np.nan, np.float32)
    py = np.full((n, len(team)), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            xy = np.array([g.split(",")[:2] for g in parts[1].split(";")], dtype=np.float32)
            px[i], py[i] = xy[:, 0] * length, xy[:, 1] * width
            b = parts[2].split(",")
            bx[i], by[i] = float(b[0]) * length, float(b[1]) * width

    period = np.zeros(n, np.int8)
    for half, (start, end) in enumerate(play_windows(game), 1):
        period[start:end + 1] = half
    return {"name": game, "px": px, "py": py, "bx": bx, "by": by, "team": team,
            "period": period, "labels": None, "pitch": (length, width)}


# ------------------------------------------------------------------ practice matches

def _csv_tracking(path):
    raw = pd.read_csv(path, skiprows=2)
    period = raw["Period"].values.astype(np.int8)
    values = raw.iloc[:, 3:].values.astype(np.float32)        # pairs x, y ... then ball x, y
    players = values[:, :-2].reshape(len(raw), -1, 2)
    ball = values[:, -2:]
    return period, players, ball


def _csv_events(path):
    ev = pd.read_csv(path)
    out = []
    for row in ev.itertuples(index=False):
        out.append({"team": 0 if row.Team == "Home" else 1, "type": str(row.Type),
                    "subtype": "" if pd.isna(row.Subtype) else str(row.Subtype),
                    "period": int(row.Period), "start": int(row._4), "end": int(row._6)})
    return out


def load_csv_game(k, controlled=False):
    folder = os.path.join(EXTERNAL_DIR, "Sample_Game_{}".format(k))
    stem = os.path.join(folder, "Sample_Game_{}_".format(k))
    period, home, ball = _csv_tracking(stem + "RawTrackingData_Home_Team.csv")
    _, away, _ = _csv_tracking(stem + "RawTrackingData_Away_Team.csv")
    players = np.concatenate([home, away], axis=1)
    team = np.r_[np.zeros(home.shape[1], np.int8), np.ones(away.shape[1], np.int8)]
    events = _csv_events(stem + "RawEventsData.csv")
    return _assemble("practice{}".format(k), players, ball, team, period, events, controlled)


def load_json_game(k=3, controlled=False):
    folder = os.path.join(EXTERNAL_DIR, "Sample_Game_{}".format(k))
    stem = os.path.join(folder, "Sample_Game_{}_".format(k))
    root = ET.parse(stem + "metadata.xml").getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of_player = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    # Which player each of the 22 slots holds changes at substitutions: one spec per range.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        chans = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        players = [player_of_channel[c] for c in chans if c.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), players))
    all_players = sorted({p for _, _, ps in specs for p in ps})
    slot = {p: i for i, p in enumerate(all_players)}
    team = np.array([team_of_player[p] for p in all_players], np.int8)

    lines = open(stem + "tracking.txt").read().splitlines()
    n = len(lines)
    players = np.full((n, len(all_players), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    spec_i = 0
    for line in lines:
        f, pl, b = line.split(":")
        f = int(f)
        while specs[spec_i][1] < f:
            spec_i += 1
        ids = specs[spec_i][2]
        xy = np.array([g.split(",") for g in pl.split(";")], dtype=np.float32)
        players[f - 1, [slot[p] for p in ids]] = xy
        ball[f - 1] = np.array(b.split(","), dtype=np.float32)

    data = json.load(open(stem + "events.json"))["data"]
    events = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub.get("name", "")
        elif isinstance(sub, list):
            sub = " ".join(s.get("name", "") for s in sub)
        else:
            sub = ""
        events.append({"team": team_ids.index(e["team"]["id"]), "type": e["type"]["name"],
                       "subtype": sub, "period": int(e["period"]),
                       "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"])})

    period = np.zeros(n, np.int8)
    for p in (1, 2):
        frames = [e["start"] for e in events if e["period"] == p]
        period[min(frames) - 1:max(e["end"] for e in events if e["period"] == p)] = p
    return _assemble("practice{}".format(k), players, ball, team, period, events, controlled)


def _assemble(name, players, ball, team, period, events, controlled=False):
    length, width = PRACTICE_PITCH
    n = len(players)
    labels, graded = labels_from_events(events, n, controlled)
    period = np.where(graded, period, 0).astype(np.int8)
    labels = np.where(period > 0, labels, NONE)
    return {"name": name, "px": players[:, :, 0] * length, "py": players[:, :, 1] * width,
            "bx": ball[:, 0] * length, "by": ball[:, 1] * width, "team": team,
            "period": period, "labels": labels, "pitch": (length, width)}


CONTROL_S = 2.0                         # "a player keeps the ball for a few seconds"
CONFIRM = {"PASS", "SHOT", "SET PIECE"}  # a completed pass / shot / restart shows control


def _controlled(est, last):
    """Drop takeovers that never became *controlled* possession (CHALLENGE.md proxies).

    Consecutive events of one team form a run. A run of the *other* team only takes the ball
    if it completes a pass, shoots, restarts play, or keeps the ball >= CONTROL_S seconds before
    the first team plays it again. Otherwise (a touch, a deflection, a won duel without control)
    possession stays with the team that had it.
    """
    runs = []
    for e in est:
        if runs and runs[-1][0]["team"] == e["team"]:
            runs[-1].append(e)
        else:
            runs.append([e])
    kept, current = [], None
    for i, run in enumerate(runs):
        nxt = runs[i + 1][0]["start"] if i + 1 < len(runs) else last + 1
        held = (nxt - run[0]["start"]) / 25.0
        if (current is None or run[0]["team"] == current
                or any(e["type"] in CONFIRM for e in run) or held >= CONTROL_S):
            kept.extend(run)
            current = run[0]["team"]
    return kept


def labels_from_events(events, n, controlled=False):
    """Per-frame H / A / D from event data, and the graded mask (kick-off to last event).

    Live ball: the team of the most recent event in which a team played the ball (pass, carry,
    shot, recovery, ball lost, set piece) -- the last team to have it keeps it while the ball is
    loose, as CHALLENGE.md says. Dead ball: from the ball going out / the foul / a goal until the
    restart (the next set piece, or the next event if none is tagged).
    """
    events = sorted(events, key=lambda e: (e["start"], e["end"]))
    labels = np.full(n, NONE, np.int8)
    graded = np.zeros(n, bool)

    for p in (1, 2):
        evp = [e for e in events if e["period"] == p]
        if not evp:
            continue
        first, last = evp[0]["start"] - 1, max(e["end"] for e in evp) - 1
        graded[first:last + 1] = True

        # 1. live possession: step function over establishing events
        est = [e for e in evp if e["type"] in ESTABLISH]
        if controlled:
            est = _controlled(est, last)
        team = evp[0]["team"]
        cursor = first
        for e in est:
            f = e["start"] - 1
            if f > cursor:
                labels[cursor:f] = team
                cursor = f
            team = e["team"]
        labels[cursor:last + 1] = team

        # 2. dead balls
        for i, e in enumerate(evp):
            sub = e["subtype"].upper()
            if e["type"] == "BALL OUT":
                stop = e["end"] - 1
            elif e["type"] == "FAULT RECEIVED":
                stop = e["start"] - 1
            elif e["type"] == "SHOT" and ("GOAL" in sub.replace("GOAL KICK", "") or "OUT" in sub):
                stop = e["end"] - 1
            else:
                continue
            restart = None
            for later in evp[i + 1:]:
                if later["start"] - 1 <= stop:
                    continue
                if later["type"] == "SET PIECE":
                    restart = later["start"] - 1
                    break
                if later["type"] in ESTABLISH and restart is None:
                    restart = later["start"] - 1
                    break
            if restart is not None and restart > stop:
                labels[stop:restart] = D
    return labels, graded


def load_practice(controlled=False):
    return [load_csv_game(1, controlled), load_csv_game(2, controlled),
            load_json_game(3, controlled)]

"""Per-frame features, from one team's point of view ("own" vs "opp").

Computing everything from both sides and training on both makes the model symmetric: it learns
what possession looks like for *a* team, and knows nothing about home or away.

Every velocity is differenced from positions (the speed column in our ATD is not available in
the practice data, so neither side uses it). Where the ball is hidden, its position is linearly
interpolated between the last and next sighting, and the model is told how long it has been
hidden -- it decides for itself how much to trust that.
"""

import numpy as np

FPS = 25
CAP_DIST = 40.0
CAP_TIME = 30.0
WINDOWS = (25, 75, 250)          # 1 s, 3 s, 10 s


def rolling_mean(x, window):
    valid = ~np.isnan(x)
    k = np.ones(window)
    total = np.convolve(np.where(valid, x, 0.0), k, "same")
    count = np.convolve(valid.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _last_next_visible(vis):
    n = len(vis)
    idx = np.arange(n)
    last = np.where(vis, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(vis, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return last, nxt


def _interp_ball(b, vis, last, nxt):
    n = len(b)
    out = b.astype(float).copy()
    hidden = ~vis
    has_last, has_next = last >= 0, nxt < n
    both = hidden & has_last & has_next
    t = (np.arange(n) - last) / np.maximum(nxt - last, 1)
    out[both] = b[last[both]] + t[both] * (b[nxt[both]] - b[last[both]])
    only_last = hidden & has_last & ~has_next
    out[only_last] = b[last[only_last]]
    only_next = hidden & ~has_last & has_next
    out[only_next] = b[nxt[only_next]]
    return out


def _velocity(p):
    v = np.full_like(p, np.nan, dtype=np.float32)
    v[1:-1] = (p[2:] - p[:-2]) * FPS / 2
    return np.clip(v, -12, 12)


def shared(match):
    """Everything that does not depend on the point of view, computed once."""
    length, width = match["pitch"]
    px, py, bx, by = match["px"], match["py"], match["bx"], match["by"]
    vis = ~np.isnan(bx)
    last, nxt = _last_next_visible(vis)
    bxi, byi = _interp_ball(bx, vis, last, nxt), _interp_ball(by, vis, last, nxt)
    n = len(bx)

    idx = np.arange(n)
    since = np.where(vis, 0, np.where(last >= 0, idx - last, CAP_TIME * FPS)) / FPS
    until = np.where(vis, 0, np.where(nxt < n, nxt - idx, CAP_TIME * FPS)) / FPS

    # ball motion from smoothed, interpolated positions
    sx, sy = rolling_mean(bxi, 5), rolling_mean(byi, 5)
    vx, vy = _velocity(sx), _velocity(sy)
    bspeed = np.hypot(vx, vy)
    lag = 3
    bdv = np.zeros(n)
    bdv[lag:-lag] = np.hypot(vx[2 * lag:] - vx[:-2 * lag], vy[2 * lag:] - vy[:-2 * lag])
    bdv = np.nan_to_num(bdv)

    out_dist = np.maximum.reduce([np.zeros(n), -bxi, bxi - length, -byi, byi - width])
    out_dist = np.where(vis, out_dist, np.nan)
    out_last = np.where(last >= 0, np.nan_to_num(out_dist)[np.maximum(last, 0)], 0)

    dist = np.hypot(px - bxi[:, None], py - byi[:, None])
    pvx, pvy = _velocity(px), _velocity(py)
    pspeed = np.hypot(pvx, pvy)
    with np.errstate(all="ignore"):
        speed_all = np.nanmean(pspeed, axis=1)

    # attacking direction of each team in each period: a team attacks away from where it
    # stands on average (its own half)
    direction = np.zeros((n, 2))
    for p in (1, 2):
        sel = match["period"] == p
        if not sel.any():
            continue
        for t in (0, 1):
            with np.errstate(all="ignore"):
                mean_x = np.nanmean(px[sel][:, match["team"] == t])
            direction[sel, t] = 1.0 if mean_x < length / 2 else -1.0

    return {"vis": vis, "last": last, "nxt": nxt, "bxi": bxi, "byi": byi, "since": since,
            "until": until, "bspeed": bspeed, "bdv": bdv, "out_dist": out_dist,
            "out_last": out_last, "dist": dist, "pvx": pvx, "pspeed": pspeed,
            "speed_all": speed_all, "direction": direction, "length": length, "width": width}


def _team_block(s, match, t):
    team = match["team"] == t
    dist = np.where(team, s["dist"], np.nan)
    with np.errstate(all="ignore"):
        srt = np.sort(np.where(np.isnan(dist), np.inf, dist), axis=1)
    d1 = np.minimum(srt[:, 0], CAP_DIST)
    d2 = np.minimum(srt[:, 1], CAP_DIST)
    n3 = np.sum(dist <= 3, axis=1)
    n8 = np.sum(dist <= 8, axis=1)
    px = np.where(team, match["px"], np.nan)
    py = np.where(team, match["py"], np.nan)
    direction = s["direction"][:, t]
    with np.errstate(all="ignore"):
        cx = (np.nanmean(px, axis=1) - s["length"] / 2) * direction
        sx = np.nanstd(px, axis=1)
        sy = np.nanstd(py, axis=1)
        vf = np.nanmean(np.where(team, s["pvx"], np.nan), axis=1) * direction
        sp = np.nanmean(np.where(team, s["pspeed"], np.nan), axis=1)
        # most advanced and deepest outfield positions in the attacking direction
        rel = (px - s["length"] / 2) * direction[:, None]
        front = np.nanmax(rel, axis=1)
    count = np.sum(~np.isnan(px), axis=1)
    return {"d1": d1, "d2": d2, "n3": n3, "n8": n8, "cx": cx, "sx": sx, "sy": sy, "vf": vf,
            "sp": sp, "front": front, "count": count, "direction": direction}


def features(match, s=None):
    """(X_home_view, X_away_view, names). Row i of each is frame i."""
    s = s or shared(match)
    blocks = [_team_block(s, match, 0), _team_block(s, match, 1)]
    views = []
    for own in (0, 1):
        o, p = blocks[own], blocks[1 - own]
        diff = np.clip(o["d1"] - p["d1"], -20, 20)
        close_o = ((o["d1"] <= 2.5) & (o["d1"] < p["d1"])).astype(float)
        close_p = ((p["d1"] <= 2.5) & (p["d1"] < o["d1"])).astype(float)
        last = np.maximum(s["last"], 0)
        nxt = np.minimum(s["nxt"], len(diff) - 1)
        cols = {
            "vis": s["vis"].astype(float), "since": np.minimum(s["since"], CAP_TIME),
            "until": np.minimum(s["until"], CAP_TIME),
            "ball_x_att": (s["bxi"] - s["length"] / 2) * o["direction"],
            "ball_y_c": np.abs(s["byi"] - s["width"] / 2),
            "out_dist": s["out_dist"], "out_last": s["out_last"],
            "bspeed": s["bspeed"], "bdv": s["bdv"],
            "d1o": o["d1"], "d1p": p["d1"], "d2o": o["d2"], "d2p": p["d2"], "diff": diff,
            "n3o": o["n3"], "n3p": p["n3"], "n8o": o["n8"], "n8p": p["n8"],
            "cxo": o["cx"], "cxp": p["cx"], "sxo": o["sx"], "syo": o["sy"],
            "sxp": p["sx"], "syp": p["sy"], "vfo": o["vf"], "vfp": p["vf"],
            "spo": o["sp"], "spp": p["sp"], "speed_all": s["speed_all"],
            "fronto": o["front"], "frontp": p["front"],
            "cnto": o["count"], "cntp": p["count"],
            "diff_last_seen": np.where(s["last"] >= 0, diff[last], np.nan),
            "diff_next_seen": np.where(s["nxt"] < len(diff), diff[nxt], np.nan),
        }
        rolled = {"diff": diff, "close_o": close_o, "close_p": close_p,
                  "speed_all": s["speed_all"], "vf_gap": o["vf"] - p["vf"],
                  "vis": s["vis"].astype(float), "bspeed": s["bspeed"],
                  "out": (np.nan_to_num(s["out_dist"]) > 1).astype(float),
                  "ball_x_att": cols["ball_x_att"], "kick_o": (s["bdv"] > 4) * close_o,
                  "kick_p": (s["bdv"] > 4) * close_p}
        for name, x in rolled.items():
            for w in WINDOWS:
                cols["{}_r{}".format(name, w)] = rolling_mean(x.astype(float), w)
        names = list(cols)
        views.append(np.column_stack([cols[k] for k in names]).astype(np.float32))
    return views[0], views[1], names

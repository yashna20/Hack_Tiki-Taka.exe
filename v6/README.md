# v6: v5's model, trained on 3 Metrica + 20 SkillCorner matches

v6 uses the same model and decoding as v5 ([v5/README.md](../v5/README.md)). The only change is
more practice matches: 20 Australian A-League matches from SkillCorner's open data.

## Why

v5 learnt stoppages from Metrica alone, where 27–35 % of the time is dead. On our match it called
only 23 % dead. SkillCorner's matches are 37–55 % dead, close to what our tracking suggests
(about 45 %), and their broadcast tracking really loses the ball about 40 % of the time, as ours
does.

## Reproduce

```bash
venv/bin/pip install -r requirements-v5.txt          # same pinned versions as v5
# Metrica sample data: see v5/README.md, step 3
sh v6/download_skillcorner.sh                        # SkillCorner, pinned commit, SHA-256 checked
venv/bin/python -m v5.measure                        # -> v5/measured.json (committed)
venv/bin/python -m v6.run experiment                 # ~25 min -> v6/params.json (committed)
venv/bin/python -m v6.run submit                     # -> evaluation/results/v6.csv
```

`submit` gives a byte-identical file on every run (checked).

## SkillCorner as practice data ([v6/skillcorner.py](skillcorner.py))

* **Tracking:** 10 fps, linearly interpolated to 25 fps.
  * **Ball:** kept only where SkillCorner detected it; NaN elsewhere, as in ours.
  * **Players:** every listed player, whether detected or extrapolated by SkillCorner's
    algorithm.
* **Labels:** SkillCorner's *phases of play*. Inside a phase, the team in possession gives H or A.
  Inside a period but outside every phase, the label is D, because phases exist only while the
  ball is in play. These are algorithmic labels on other games.
* **No artificial degradation:** this tracking is already broadcast-quality. The Metrica matches
  are degraded exactly as in v5.

## Practice test

All 23 matches are split into 4 fixed folds, and each fold is held out once. Two models train on
the rest: one on Metrica only (v5's data) and one on Metrica + SkillCorner (v6). Switch penalties
come from the same grid for both. The score is the mean of the Metrica mean and the SkillCorner
mean.

| method | Metrica spellF1 / macroF1 | SkillCorner spellF1 / macroF1 | combined |
|---|---|---|---|
| baseline_smooth50 | 0.339 / 0.692 | 0.352 / 0.735 | 0.346 |
| v5-style (Metrica only) | 0.753 / 0.897 | 0.521 / 0.730 | 0.637 |
| **v6** (Metrica + SkillCorner) | 0.723 / 0.898 | **0.687 / 0.909** | **0.705** |

v6 beats v5-style by +0.068, above the 0.04 noise bar, so v6 is used. Penalties: 12 / 24.

On Barça–Athletic, v6 labels 37 % of frames dead (v5: 23 %) and finds 246 possessions, with a
mean of 16.3 s and a median of 9.8 s.

## Caveat

SkillCorner's possession labels come from their own algorithm, not from a human reading of
"controlled possession". The baseline scores 0.35 on them against 0.243 on the real match, so
SkillCorner is not a harder test than Metrica. Its value is realistic stoppages and real ball
loss.

## Data and credits

* **SkillCorner Open Data:** <https://github.com/SkillCorner/opendata>, MIT licence
  (© 2020 SkillCorner), commit `4340d27`. Downloaded to `data/external/` (gitignored), not
  redistributed.
* **Metrica Sports sample data:** see [v5/README.md](../v5/README.md).

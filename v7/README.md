# v7: final submission

v7 has the same model and decoding as v5/v6, trained on 3 Metrica + 20 SkillCorner matches. All
of them are labelled with the challenge's **controlled-possession** rule.

## Reproduce `evaluation/results/v7.csv`

```bash
python3.14 -m venv venv && venv/bin/pip install -r requirements-v5.txt   # pinned versions

# practice data (both go to data/external/, which is gitignored)
git clone https://github.com/metrica-sports/sample-data data/external/sample-data
git -C data/external/sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996
sh v6/download_skillcorner.sh

venv/bin/python -m v5.measure            # optional: regenerates v5/measured.json (committed)
venv/bin/python -m v7.run experiment     # optional, ~35 min: regenerates v7/params.json (committed)
venv/bin/python -m v7.run submit         # -> evaluation/results/v7.csv
venv/bin/python evaluation/validate_submission.py evaluation/results/v7.csv
```

`submit` gives a byte-identical file on every run (checked).

## The pipeline

1. **Practice matches** (other games only):
   * 3 Metrica sample matches ([v4/data.py](../v4/data.py)), with tracking degraded to look like
     ours ([v4/degrade.py](../v4/degrade.py), using numbers from [v5/measure.py](../v5/measure.py)).
   * 20 SkillCorner A-League matches ([v6/skillcorner.py](../v6/skillcorner.py)), broadcast
     tracking used as it is.
2. **Labels**, with one rule for all 23 matches: the other team only takes the ball after a
   completed pass, a shot, a restart, or at least 2 s on the ball (CHALLENGE.md's proxies). A
   touch or deflection doesn't count.
   * Metrica: from its event data.
   * SkillCorner: from its player-possession events ([v7/sc_controlled.py](sc_controlled.py)).
     Dead ball means outside SkillCorner's phases of play.
3. **Features:** about 70 measurements per frame from each team's point of view
   ([v4/features.py](../v4/features.py)): distances to the ball, players near it, team shape
   and speed, ball outside the pitch, time since the ball was last seen, and rolling means over
   1 / 3 / 10 s.
4. **Model:** gradient-boosted trees give P(own team / other team / dead). Both points of view
   are averaged into P(H), P(A), P(D).
5. **Decoding:** Viterbi with switch penalties. H↔A costs 16 and live↔dead costs 24; both were
   chosen on the practice test.
6. **Fixed rules:** second-half graded frames before the kick-off (anchor in `games.py`) are D;
   frames outside the halves are X.

## Practice test

Each match is held out once (4 fixed folds) and labelled by a model that never saw it.
Scores are spellF1 / macroF1.

| | Metrica (3) | SkillCorner, controlled labels (20) | SkillCorner, phase labels (20) |
|---|---|---|---|
| baseline_smooth50 | 0.339 / 0.692 | 0.355 / 0.732 | 0.352 / 0.735 |
| v5-style (Metrica only) | 0.753 / 0.897 | – | 0.521 / 0.730 |
| v6 | 0.723 / 0.898 | 0.685 / 0.902 | 0.687 / 0.909 |
| **v7** | **0.733** / 0.898 | **0.698** / 0.905 | 0.673 / 0.905 |

**Choosing v7.**
* v7 and v6 are within noise of each other (±0.015).
* v7 is submitted because all of its training labels follow the challenge's definition.
* Both are far more robust than v5 on broadcast-quality tracking (0.67–0.70 against 0.52).
* The public 5 minutes were used only to check the file format.

**On Barça–Athletic:** 37 % of frames are labelled dead and 245 possessions are found, with a
mean of 16.2 s and a median of 10.1 s.

## Known limitations

* Practice scores are optimistic: the baseline scores 0.34–0.36 on practice but 0.243 on the
  real match.
* SkillCorner's controlled labels are built from its algorithmic possession events, not from
  human annotation.

## Data and credits

* [Metrica Sports sample data](https://github.com/metrica-sports/sample-data), commit
  `e706dd5`. The repository has no explicit licence file; it is credited here.
* [SkillCorner Open Data](https://github.com/SkillCorner/opendata), MIT licence, commit `4340d27`.
* Neither dataset, nor the competition's match data, is in this repository.

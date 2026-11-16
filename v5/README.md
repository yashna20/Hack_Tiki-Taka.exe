# v5 — learned possession model + sticky-possession decoding

v5 labels every frame of Barça–Athletic as `H`, `A`, `D` or `X`. A gradient-boosted tree model,
trained **only on other matches**, gives the chance of home / away / dead ball in every frame. A
Viterbi decoder then turns those chances into whole possessions: switching labels costs a penalty,
so one touch doesn't flip possession.

## Reproduce the submission

```bash
# 1. environment (Python 3.14.0, exact versions)
python3.14 -m venv venv
venv/bin/pip install -r requirements-v5.txt

# 2. the provided match data, unpacked into data/ as described in data/README.md

# 3. the practice matches (Metrica Sports sample data), at the exact commit used
git clone https://github.com/metrica-sports/sample-data data/external/sample-data
git -C data/external/sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996

# 4. measure our tracking -> v5/measured.json (already committed; this regenerates it)
venv/bin/python -m v5.measure

# 5. practice test, picks every setting -> v5/params.json (already committed; ~8 min)
venv/bin/python -m v5.run experiment

# 6. the submission -> evaluation/results/v5.csv
venv/bin/python -m v5.run submit
venv/bin/python evaluation/validate_submission.py evaluation/results/v5.csv
```

Step 6 alone regenerates the submission from the committed `measured.json` and `params.json`.
Running it twice gives byte-identical files (checked).

Neither the match data nor the practice data are in this repository (`data/*` is gitignored).
`v5/measured.json` holds only summary statistics, such as percentages and gap-length percentiles,
from which the tracking can't be reconstructed.

## What each file does

| file | role |
|---|---|
| `v4/data.py` | loads our match and the 3 practice matches into one format; turns the practice event data into frame labels |
| `v4/degrade.py` | makes the practice tracking look like ours: the ball goes missing, appears outside the pitch at stoppages, players drop out |
| `v4/features.py` | about 70 measurements per frame, from each team's point of view |
| `v5/measure.py` | measures our tracking (ball-missing rates, gap lengths, ball outside the pitch) → `measured.json` |
| `v5/run.py` | practice test (`experiment`) and the submission (`submit`) |
| `v5/params.json` | the settings the practice test chose |

## How every setting was chosen

Nothing was tuned on Barça–Athletic's ground truth, the public 5 minutes included.

* **Practice labels** come mechanically from Metrica's human event data on *other* matches. A
  change of possession counts only once it is **controlled**: the new team completes a pass,
  shoots, restarts play, or keeps the ball for at least 2 s. These are the proxies listed in
  CHALLENGE.md. Ball out, foul and goal start a stoppage that runs until the restart.
* **Making practice look like ours** uses numbers measured by `v5/measure.py` on our *tracking
  only*. Live and dead frames are split with the algorithmic labels of `starter/v2.py`. v5's own
  addition is that live play hides the ball only in short bursts (at most about 12 s), as in our
  live play.
* **Model settings, switch penalties, and whether to add v2's rule-based scores** were chosen by
  leave-one-match-out testing on the 3 practice matches (train on 2, test on the 3rd, rotate, 2
  random degradations each). v2's scores gained +0.004 spellF1, under the 0.04 noise bar, so
  they are off (`w = 0`).
* **Fixed rules**: graded frames of the second half before the kick-off (from the anchor in
  `games.py`) are `D`; frames outside the two halves are `X`.

### Why v5 is submitted and not v4

Both score 0.752 spellF1 on the practice test. v5 is chosen because its practice labels follow
the challenge's *controlled possession* definition, which v4's don't. The public 5 minutes were
used only to check the file format, not to choose between them.

## Results

| | practice spellF1 | practice macroF1 | full match |
|---|---|---|---|
| baseline_smooth50 | 0.335 | 0.692 | 0.243 (published) |
| v5 | 0.752 | 0.895 | 303 possessions, 23 % dead |

The practice test is easier than the real match: the baseline scores 0.335 there and 0.243 on
the full match. So read practice scores only as a comparison between methods.

## Known limitations

* Only 3 practice matches.
* v5 labels 23 % of the match dead, probably fewer than real.
* Two constants shared with v4 (`OUT_BEYOND` in `v4/degrade.py`) differ slightly from today's
  measurement: 1.3/3.0/6.3 m in the code against 1.0/2.7/5.8 m measured. They are kept so that
  v4 stays reproducible; `v5/measure.py` prints both.

## Data and credits

* **Practice matches:** [Metrica Sports sample data](https://github.com/metrica-sports/sample-data),
  3 anonymised matches with tracking and event data, commit `e706dd5`. The repository carries no
  explicit licence file; it is published openly by Metrica Sports for public use, and we credit
  it here.
* **Match data:** provided by the organisers for this competition only; not redistributed.

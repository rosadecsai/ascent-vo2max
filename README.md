# ASCENT — towards a terrain-invariant VO2max estimator for trail and mountain running

Code accompanying the preprint *"Terrain dependence of heart-rate-based VO2max estimation: mechanisms,
a within-runner field test on 9,962 runs, and a terrain-robust estimator"* (J. A. García and
R. Rodriguez-Sánchez, University of Granada, October 2026; `report/arxiv.tex`, PDF in `report/arxiv.pdf`).

ASCENT (Altitude-, Slope- and Cardiac-kinetics-corrected Estimation with uNcertainty-weighted
Tracking) estimates VO2max from the signals a GPS sports watch records (time, heart rate, GPS
distance, barometric altitude, cadence) plus the user's resting and maximal heart rate. It
models gait-specific grade cost on the 3-D path, first-order HR kinetics, an HR-equivalent
descent cost, altitude and cardiac drift, weights every minute by an explicit error budget,
and tracks fitness across sessions with a (race-aware) Kalman filter.

## Layout

```
ascent/
  physiology.py     cost models (Minetti 2002 rescaled to ACSM), Strava-like HR grade factor, altitude
  simulate.py       data-generating model: courses, runners, pacing, physiology, sensors
  estimators.py     Garmin/Firstbeat-like reconstructions, ASCENT, Kalman / race-aware trackers
  strava.py         loader for the case-study Strava streams
  fitrec.py         loader for the FitRec extract (tools/filtrar_fitrec.py)
  io.py             readers for .fit (needs fitdecode), .tcx, .gpx and Strava-stream .json files
  cli.py            command-line tool: analyse exported activities, keep a tracked value
experiments/
  run_sim.py        main comparison, ablation, sensitivity, longitudinal, structural mismatch
  attribution.py    mechanism attribution (switch one mechanism off at a time)
  run_real.py       case study on Strava activities (frozen ASCENT configuration)
  real_sensitivity.py  case study: confidence intervals, terrain-prior and HRmax sensitivity
  figures.py        all figures;   make_tables.py   LaTeX tables + number macros
  fitrec_describe.py, fitrec_run.py, fitrec_analysis.py, fitrec_diag.py, fitrec_figures.py
                    terrain-invariance test on the FitRec (Endomondo) dataset
  common.py, figstyle.py
tests/              sanity tests (physiology values, recovery in idealised worlds, file readers)
results/            result files (pickles/CSVs) and frozen_ascent_config.json
tools/              filtrar_fitrec.py (FitRec extraction) and session-specific helpers
report/             preprint LaTeX source (arxiv.tex), number macros, tables, figures, PDF
```

## Versions

- **v1**: the configuration frozen before the case study (report Sections 4-8).
- **v2** (default in the command-line tool; `es.ascent_v2`): fixed 55-s HR lag, climbing-based
  terrain prior when the surface is unknown (`surface_class="auto"`), tighter drift prior, and no
  gait labels without cadence. Developed on half of the Endomondo runners and confirmed on the other
  half, the simulation and the case study (report Section 9). `es.ascent(...)` with default
  `AscentConfig()` is still v1, so every earlier result reproduces.

## Run it on your own activities

Export the activity from Garmin Connect (gear icon → *Export Original* gives a .zip with the
.fit file, which can be passed as is; *Export to TCX* / *Export to GPX* also work) and run, from
this folder (the one that contains `ascent/`):

```
pip install numpy scipy pandas fitdecode          # fitdecode only for .fit files
python3 -m ascent.cli 20441234567.zip --hr-rest 52 --hr-max 170 --history my_vo2max.csv
python3 -m ascent.cli race.fit --hr-rest 52 --hr-max 170 --race --history my_vo2max.csv
```

For each file it prints the ASCENT estimate with its uncertainty, the fitted HR lag and drift,
where the information came from (climbs, flats, descents, hiking) and the four Garmin-like
reconstructions. `--history` accumulates sessions in a CSV file and prints the value of the
race-aware Kalman tracker (mark races with `--race`, or set `race` to 1 in the CSV later).
The surface (road/trail) is read from .fit files ("Trail Run") and Garmin .gpx files; for other
files pass `--surface trail` or `--surface road` (otherwise v2 infers a terrain prior from the
climbing of the run, which misjudges hilly roads and flat trails). `--v1` runs the original
configuration. If the tool notes that recorded HR exceeds your
HRmax setting, fix HRmax first: every HR-based estimate depends on it.

## Quick use from Python

```python
from ascent import estimators as es
p = es.prep(t, hr, dist, alt, cad)                  # 1-Hz resampling and derived signals
res = es.ascent_v2(p, hr_rest=52, hr_max=170, surface_class="trail")   # es.ascent(...) = v1
print(res["vo2max"], res["sd"], res["tau"], res["drift"], res["weight_share"])

trk = es.RaceAwareTracker()                         # displayed value across sessions
for day, value, sd, is_race in sessions:
    shown = trk.update(day, value, sd, race=is_race)
```

## Reproducing the report

```
python3 -m pytest -q tests
cd experiments
python3 run_sim.py all                 # ~17 min on 2 cores (main, ablation, sensitivity, longitudinal)
python3 -c "import run_sim; run_sim.run_extra('ablation_matched', 150); run_sim.run_mismatch(100)"   # ~11 min
python3 attribution.py 60              # ~3 min
python3 run_real.py && python3 real_sensitivity.py      # case study on the authors' own activities (data not public; not part of the preprint)
python3 figures.py main example attribution ablation sensitivity longitudinal   # add "real" only with the case-study data
python3 make_tables.py
# FitRec terrain-invariance test (download endomondoHR.json.gz from the FitRec page first, ~2 GB):
python3 ../tools/filtrar_fitrec.py /path/to/endomondoHR.json.gz   # writes fitrec_runs_*.npz + index (move to data/fitrec/)
python3 fitrec_describe.py && python3 fitrec_run.py && python3 fitrec_analysis.py && python3 fitrec_diag.py && python3 fitrec_figures.py
# ASCENT v2 (development half, test half, simulation replay, case study):
python3 fitrec_v2.py dev && python3 fitrec_v2.py test v1 TAD && python3 sim_v2.py main 300 && python3 sim_v2.py mismatch 100 && python3 real_v2.py
cd ../report && pdflatex arxiv.tex && pdflatex arxiv.tex && pdflatex arxiv.tex   # 3 passes for the contents
# numbers.tex and tables/ are committed, so the PDF rebuilds without re-running the experiments
# (results/*.pkl are not in the repository; the CSV summaries are).
```

Seeds are fixed. Requirements: Python ≥ 3.10, numpy, scipy, pandas, matplotlib (pytest for tests;
fitdecode to read .fit files).

## Caveats

The "Garmin-like" estimators are reconstructions of a proprietary method from its public
description, not Garmin's code. The simulation encodes literature-based assumptions (see the
report, Section 5 and Table 1). The case study involves one athlete and is statistically
inconclusive for terrain comparisons in training sessions.

`data/fitrec/` (the FitRec extract) is not included: the dataset's authors ask that it is not
redistributed (academic use only; cite Ni, Muhlstein & McAuley, WWW 2019).

`data/strava/` (the authors' own activities, used only in an internal case study) is not part of
the repository; `run_real.py`, `real_sensitivity.py` and `real_v2.py` need it and are kept for
completeness.

## License

MIT (see `LICENSE`). If you use this code, please cite the preprint.

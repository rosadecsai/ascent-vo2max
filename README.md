# ASCENT — terrain dependence of heart-rate-based VO2max estimates

Code accompanying the article *"Terrain dependence of heart-rate-based VO2max estimates: a
laboratory-free within-runner field test on 9962 runs, simulated mechanisms and a proof-of-concept
estimator"* (J. A. García, R. Rodriguez-Sánchez and J. Fdez-Valdivia, University of Granada),
under review, and the earlier preprint (`report/arxiv.tex`, `report/arxiv.pdf`).

**Releases.** `v1.0.0` is the code of the original submission. `v1.1.0` is the code of the first
revision: the estimators and their settings are identical; the revision adds analyses (bootstrap
intervals, counts, comparators, common set, heart-rate anchors, sample spacing, decomposition of
v2, equivalence), redrawn figures and new tables.

**What the code does and does not show.** The field test compares the estimate from a run with
the estimates from the same runner's flat runs. It measures whether an estimator depends on
terrain; it does not measure accuracy, because no measured VO2max is available. The four
"reconstructions" follow the publicly described heart-rate–speed principle with settings that are
our own choices; they are not any manufacturer's algorithm. ASCENT is a proof of concept, not a
validated estimator. The preprint is an earlier text whose stronger wording the article withdraws.

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
  estimators.py     reconstructions of the HR-speed method, ASCENT (v1, v2), Kalman / race-aware trackers
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
  fitrec_describe.py, fitrec_run.py, fitrec_analysis.py, fitrec_diag.py, fitrec_v2.py
                    within-runner field test on the FitRec (Endomondo) dataset (original submission)
  fitrec_revision.py, fitrec_revision_analysis.py
                    analyses added in revision: bootstrap intervals, counts, comparators, common
                    set, heart-rate anchors, sample spacing, decomposition of v2, equivalence
  sim_v2.py, sim_sampling.py
                    simulation replay of v2, and of the field recording conditions
  fitrec_figures.py, make_tables_rev.py
                    figure and tables of the revised article (numbers_rev.tex, tables/tab_r1_*.tex)
  common.py, figstyle.py
tests/              sanity tests (physiology values, recovery in idealised worlds, file readers)
results/            result files (pickles/CSVs) and frozen_ascent_config.json
tools/              filtrar_fitrec.py (FitRec extraction) and session-specific helpers
report/             preprint (arxiv.tex, arxiv.pdf), number macros (numbers.tex, numbers_rev.tex), tables, figures
```

## Versions

- **v1**: the configuration frozen before the case study (report Sections 4-8).
- **v2** (default in the command-line tool; `es.ascent_v2`): fixed 55-s HR lag, climbing-based
  terrain prior when the surface is unknown (`surface_class="auto"`), tighter drift prior, and no
  gait labels without cadence. Chosen on half of the Endomondo runners and evaluated once on the
  other half, which is not a fully independent sample (the diagnosis that motivated the changes
  used runs of all runners). Most of its effect on the field test comes from the climbing-based
  prior, a correction tied to the same variable that defines the classes of the test.
  `es.ascent(...)` with default `AscentConfig()` is still v1, so every earlier result reproduces.

## Run it on your own activities

Export the activity from Garmin Connect (gear icon → *Export Original* gives a .zip with the
.fit file, which can be passed as is; *Export to TCX* / *Export to GPX* also work) and run, from
this folder (the one that contains `ascent/`):

```
pip install numpy scipy pandas fitdecode          # fitdecode only for .fit files
python3 -m ascent.cli 20441234567.zip --hr-rest 50 --hr-max 185 --history my_vo2max.csv
python3 -m ascent.cli race.fit --hr-rest 50 --hr-max 185 --race --history my_vo2max.csv
```

For each file it prints the ASCENT estimate with its uncertainty, the fitted HR lag and drift,
where the information came from (climbs, flats, descents, hiking) and the four
reconstructions (the columns of the history file keep their former names, `garmin_like_*`). `--history` accumulates sessions in a CSV file and prints the value of the
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
res = es.ascent_v2(p, hr_rest=50, hr_max=185, surface_class="trail")   # es.ascent(...) = v1
print(res["vo2max"], res["sd"], res["tau"], res["drift"], res["weight_share"])

trk = es.RaceAwareTracker()                         # displayed value across sessions
for day, value, sd, is_race in sessions:
    shown = trk.update(day, value, sd, race=is_race)
```

## Reproducing the results

```
python3 -m pytest -q tests
cd experiments
python3 run_sim.py all                 # ~17 min on 2 cores (main, ablation, sensitivity, longitudinal)
python3 -c "import run_sim; run_sim.run_extra('ablation_matched', 150); run_sim.run_mismatch(100)"   # ~11 min
python3 attribution.py 60              # ~3 min
python3 run_real.py && python3 real_sensitivity.py      # case study on the authors' own activities (data not public; not part of the preprint)
python3 figures.py main example attribution ablation sensitivity longitudinal   # add "real" only with the case-study data
python3 make_tables.py
# FitRec within-runner field test (download endomondoHR.json.gz from the FitRec page first, ~2 GB):
python3 ../tools/filtrar_fitrec.py /path/to/endomondoHR.json.gz   # writes fitrec_runs_*.npz + index (move to data/fitrec/)
python3 fitrec_describe.py && python3 fitrec_run.py && python3 fitrec_analysis.py && python3 fitrec_diag.py && python3 fitrec_figures.py
# ASCENT v2 (development half, test half, simulation replay, case study):
python3 fitrec_v2.py dev && python3 fitrec_v2.py test v1 TAD && python3 sim_v2.py main 300 && python3 sim_v2.py mismatch 100 && python3 real_v2.py
# Analyses, figure and tables of the revised article (about 25 min on 2 cores):
python3 fitrec_revision.py && python3 fitrec_revision_analysis.py && python3 sim_sampling.py
python3 fitrec_figures.py && python3 make_tables_rev.py
cd ../report && pdflatex arxiv.tex && pdflatex arxiv.tex && pdflatex arxiv.tex   # 3 passes for the contents
# numbers.tex and tables/ are committed, so the PDF rebuilds without re-running the experiments
# (results/*.pkl are not in the repository; the CSV summaries are).
```

Seeds are fixed. Requirements: Python ≥ 3.10, numpy, scipy, pandas, matplotlib (pytest for tests;
fitdecode to read .fit files).

## Caveats

The reconstructions (`es.reconstruction`, formerly `es.garmin_like`, which remains as an alias)
follow the public description of a proprietary method; no numerical setting of any production
algorithm is public, and every setting here is our own choice. The simulation encodes
literature-based assumptions and shares functional forms with ASCENT, so its results are
conditional on them. No VO2max was measured: the field test shows dependence on terrain, not
accuracy.

`data/fitrec/` (the FitRec extract) is not included: the dataset's authors ask that it is not
redistributed (academic use only; cite Ni, Muhlstein & McAuley, WWW 2019).

`data/strava/` (the authors' own activities, used only in an internal case study) is not part of
the repository; `run_real.py`, `real_sensitivity.py` and `real_v2.py` need it and are kept for
completeness.

## License

MIT (see `LICENSE`). If you use this code, please cite the article (see `CITATION.cff`).

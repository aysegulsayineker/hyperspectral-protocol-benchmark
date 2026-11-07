# Buffered block evaluation for hyperspectral image classification

Code, run-level results, generated tables and figures for a study of how the
evaluation protocol, the training budget and the label budget decide the
ranking of hyperspectral classifiers. The manuscript itself is under review and
is not part of this repository.

## What the study measures

Patch-based models are normally validated by drawing training and test pixels
at random from one scene. Because each pixel is classified from a window of its
neighbours, that places training pixels inside test windows: across the four
benchmark scenes used here, 91.6 to 99.96 per cent of test windows contain at
least one training pixel at a patch side of 11. This repository contains the
audit that measures it, a buffered block split that drives the count to exactly
zero, an unbuffered block control that separates the buffer from the block
geography, and the runs that compare seven models under all three protocols.

Headline numbers, all reproducible from the files here:

| | random | blocks, no buffer | buffered blocks |
|---|---|---|---|
| 2-D CNN | 0.9894 | 0.8904 | 0.8417 |
| SSFTT | 0.9960 | 0.9038 | 0.8330 |
| LSSA | 0.9932 | 0.8869 | 0.8215 |
| 3-D CNN | 0.9909 | 0.8843 | 0.8210 |
| HybridSN | 0.9874 | 0.8719 | 0.8034 |
| ASCN | 0.9505 | 0.8596 | 0.8024 |
| spectral SVM | 0.9304 | 0.8410 | 0.7794 |

Mean overall accuracy over four scenes and five seeds. The 2-D CNN is fourth of
seven under the conventional protocol and first under the buffered split.

## Layout

```
src/            protocol, models, training and the table generator
  hsidata.py        scene loading, the three splits, the overlap audit
  models.py         spectral SVM, 2-D CNN, 3-D CNN, HybridSN, SSFTT
  proposed.py       ASCN and LSSA, the two architectures we designed
  train.py          training, metrics, protocol budget matching
  run_grid.py       the protocol cross
  run_mechanism.py  window, capacity, training budget, low label
  make_paper_tables.py  every table, straight from the result CSVs
  make_figures.py       both figures, from the same CSVs
results/        run-level CSVs, one row per run
tables/         the eleven generated tables, CSV and Markdown
figures/        the two figures
notebooks/      the Kaggle notebook used to run the grids on a single T4
protocol_config.json  block sides and the pinned class sets
```

## Reproducing

The scenes are not redistributed here. Pavia University, Pavia Centre, Salinas,
Indian Pines, Kennedy Space Center and Botswana come from the hyperspectral
scenes archive of the Computational Intelligence Group at the University of the
Basque Country; Houston 2013 comes from the IEEE GRSS Data Fusion Contest.
Place the `.mat` files in `data/` under the names `hsidata.load_cube` expects.

```
python src/run_grid.py          # protocol cross, 420 runs
python src/run_mechanism.py     # window, capacity, budget, low label, 1,320 runs
python src/make_paper_tables.py # regenerates tables/ from results/
python src/make_figures.py      # regenerates figures/ from results/ and tables/
```

No number in the tables is typed by hand; they are generated from the CSVs in
`results/` by a single script.

## Notes on reading the results

* The protocol is called a **buffered block split**, not a leakage-free split.
  The buffer removes patch overlap between splits; it does not remove spatial
  autocorrelation at longer range.
* The contrast between the buffered split and the unbuffered control is an
  **operational** one. Dropping the buffer returns the buffer-ring pixels as
  well as the overlapping windows: 59 to 80 per cent of the control's test
  pixels are pixels the buffered split had discarded (`results/ring_confound.csv`).
  It is not a causal estimate of patch-overlap leakage.
* The 20 scene-by-seed cells are not independent. Every claim is therefore also
  checked with the scene as the unit, in `tables/table6_scene_cluster.csv`.

## Licence

Code is released under the MIT licence (see `LICENSE`). The result CSVs,
tables and figures are released under CC BY 4.0.

# IMPROVEMENTS.md

*Analysis date: 2026-07-11.*

Superwelch is a research/teaching project exploring whether machine-learned classifiers (random forests, NNs) can outperform Welch's t-test at deciding "did this experiment do anything?" for small-sample A/B experiments. It comprises `simulator.py` (a CLI that generates synthetic experiments, trains a RandomForest, and logs results to `simulation_database.sqlite`), several exploratory Jupyter notebooks (`superwelch-eda`, `-nn`, `-nextgen`, `-hailmary`), a Lazarus/FreePascal GUI in `gui/`, and a Stream Deck demo app in `demo/`. Current state: mid-experiment. The last commit (c79729c) explicitly says "I should save the profiles too" — and indeed the entire `demo/` payload (`main.py`, `assets/`, `tones/`, `train/`, the `.streamDeckProfilesBackup`), `gui/`, `diagrams.ipynb`, and `poset-based testing.ipynb` are all untracked, and four notebooks have uncommitted modifications.

## Bugs & Fixes

- **Finish the last commit's stated intent.** c79729c says the Stream Deck profiles should be saved; `demo/Stream Deck - 29-10-2023 - 18-38.streamDeckProfilesBackup` and the two `manifest-with-profiles`/`manifest-without-profiles` files under `demo/src/au.org.ifost.superwelchdemo.sdPlugin/` are still untracked. Commit them (or record why not) before the context evaporates.
- **`simulator.py` seeding is incomplete.** `numpy.random.seed(args.experimental_data_rng_seed)` is set, but `generate_experiment()` draws via `scipy.stats.*.rvs()` without a `random_state`; that works only because scipy falls back to the legacy global numpy RNG. Modern practice: pass a `numpy.random.Generator` explicitly so results stay reproducible across numpy/scipy upgrades. Also verify `--rfc-seed` is actually threaded into the `RandomForestClassifier(random_state=...)` — the help text ends mid-word ("classific"), which smells like an unfinished edit.
- **No SQLite schema management.** `simulator.py` opens `simulation_database.sqlite` and assumes tables exist. Add `CREATE TABLE IF NOT EXISTS` so a fresh clone can run `wrapper.sh` without a mystery crash.

## Improvements

- **Commit or ignore the working tree.** Decide what `gui/`, `demo/`, `diagrams.ipynb`, and `poset-based testing.ipynb` are: real work → commit; scratch → `.gitignore`. Right now a `git clone` loses roughly half the project.
- **Add a `.gitignore`.** At minimum: `.DS_Store`, `demo/tmp/`, `*~`, `#*#` (Emacs droppings like `demo/#manifest.json#` and `main.py~` are currently polluting the tree — commit message 497cd1d "The bane of my existence" suggests this fight is ongoing; `.gitignore` ends it), `simulation_database.sqlite` if it's regenerable.
- **Extract shared logic from the notebooks.** The feature-engineering ("Sort the values, because that's what the ML model was trained on" — 191cf67) lives in both `simulator.py` and notebooks; sorting-before-inference is a correctness invariant that should be one function in a small module imported by both, not duplicated.
- **`wrapper.sh` runs 675 sequential simulations.** Add `xargs -P`/GNU parallel or at least `set -e` and a resume mechanism keyed on the DB (skip already-completed tag/seed/size combos).

## Testing

- There are zero tests. Highest-value first test: a smoke test that `simulator.py --dry-run --number-of-training-experiments 100 --number-of-testing-experiments 10 --tag test` exits 0. Second: a regression test that with a fixed seed the RF accuracy matches a recorded value, protecting the reproducibility claims the whole research rests on.
- "poset-based testing.ipynb" hints at a testing idea — promote it out of a notebook if it's real.

## Documentation

- **No top-level README.** The repo's thesis (ML vs Welch's t-test for tiny samples) is only reconstructable from notebook archaeology. Write a README covering: the research question, what each notebook concluded (BMAP3, the "sneaky" prior-informed variant from 905f689, the RF variant from 64d42d2), how to run `simulator.py`/`wrapper.sh`, and what the Stream Deck demo demonstrates.
- `demo/README.md` exists but is untracked; commit it.

## Security

- No committed secrets spotted. `demo/requirements.txt` pins `certifi==2022.12.7` and `requests==2.28.2`, both of which have known CVEs (certifi e-Tugra removal, requests CVE-2023-32681 proxy header leak); this is another argument for the uv migration below, which will pull current versions.

## Housekeeping / Modernization

- **Migrate off `requirements.txt` to uv.** `demo/requirements.txt` (and its `~` backup) should go: create a `pyproject.toml` with `uv init`, `uv add streamdeck-sdk requests ...`, commit `pyproject.toml` + `uv.lock`, delete `requirements.txt`, and run everything via `uv run demo/main.py` and `uv run simulator.py`. This also unpins pydantic 1.10 / streamdeck-sdk 0.3.1, which are from 2023.
- `simulator.py`'s pattern of argparse-before-imports is fine for CLI snappiness, but move the sqlite connect out of module scope so the file is importable by tests.
- Delete or ignore `demo/tmp/` and all `*~`/`#*#` files.

## Quick Wins

1. `git add` the demo profiles + `demo/README.md` and finish commit c79729c's intent.
2. Add `.gitignore` (.DS_Store, `*~`, `#*#`, `demo/tmp/`).
3. Fix the truncated `--rfc-seed` help string and confirm the seed is wired to the RF.
4. Commit the four dirty notebooks (or `nbstripout` them first to shrink diffs).
5. Write a five-paragraph README.

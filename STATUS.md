# STATUS

## Phase 0: Scaffold and configuration - COMPLETE

Date: 2026-11-09
Specification: Architecture V7 (approved)

### Completed

- Repository scaffold created: package layout under `src/urbanflood/`, test
  suite under `tests/`, pytest bootstrap via `conftest.py` (no installation
  step needed), `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`
  with the prominent SYNTHETIC banner.
- `src/urbanflood/config.py`: full locked V7 configuration as frozen,
  validating dataclasses:
  - grid: 96 x 96 cells at 50 m (cell area 2500 m2, domain 4.8 km);
  - vertical datum MSL = 0.0 m, terrain [0.2, 60.0] m, coastal band pin
    [0.2, 1.2] m, deck freeboard 1.0 m, outfall freeboard 0.0 m;
  - seeded streams: master seed 2027, independent named streams (env,
    drainage, weather, routing, od_pairs) via SHA-256 payload derivation in
    `rng.py`; 5 Experiment A environment seeds as env sub-streams;
  - rainfall: exact pulse timings t_s = 3600 s, t_p = 8400 s, t_e = 13200 s
    derived from peak/total (R: 90 mm/h, 120 mm; S1: 150 mm/h, 200 mm) with
    triangle-consistency validation; V_precip / V_runoff / V_loss partition
    documented on the relevant classes;
  - tide: T = 44712 s, zero phase, zero surf, analytical eta() helper,
    peak-at-T/4 property;
  - drainage: capacity 20-80 mm/h, efficiency 0.5-1.0, blocked fractions
    0.20 / 0.40, outfall rule Z_out = z_i;
  - exchange: c = 0.1 m/s empirical, dt = 30 s default and 15 s sensitivity,
    film threshold 0.005 m, f_step = 0.12 at default with the declared
    0.20 design bound enforced on assembly;
  - susceptibility: declared ranges and the 0.35/0.25/0.15/0.15/0.10 weights
    with sum-to-one validation;
  - routing: closure 0.30 m with sweep {0.20, 0.30, 0.45}, label 0.10 m with
    sweep {0.05, 0.10, 0.20, 0.30}, lambda sweep {0, 0.5, 1, 2, 5, 10},
    speeds 50/30/20 km/h, 20 OD pairs, 3-point edge sampling;
  - scenarios: locked R/S1/S2/S3 one-factor-at-a-time set with R as reference;
  - synchronous-update and canonical-ordering conventions documented in the
    module docstring and in README.md.
- `src/urbanflood/rng.py`: versioned stream payload, deterministic and
  reproducible derivation, lazy numpy generator binding, stream-name export.

### Changed files

Created:
- README.md, STATUS.md, requirements.txt, pyproject.toml, .gitignore
- conftest.py
- src/urbanflood/__init__.py
- src/urbanflood/rng.py
- src/urbanflood/config.py
- tests/test_config.py
- tests/test_rng.py
- tests/test_ascii.py

No pre-existing files were overwritten or deleted (directory was empty at
Phase 0 start; verified).

### Test results (actually executed, final run)

Command: `python -m pytest -v`
Result: 34 passed, 0 failed, 0 skipped in 0.14 s
Platform: win32, Python 3.13.5, pytest 8.3.4

Breakdown:
- tests/test_ascii.py: 1 passed (TC6 ASCII-only scan across 12 files; em dash,
  en dash, and all non-ASCII bytes rejected)
- tests/test_config.py: 24 passed (locked values, validation rejections,
  frozen immutability, scenario structure)
- tests/test_rng.py: 9 passed (determinism, independent reimplementation,
  static golden values, stream distinctness, master-seed sensitivity,
  numpy generator reproducibility)

First run: 30 passed / 4 failed; both runs' outputs were captured. Failures
resolved before commit:

1. ElevationConfig validation required coastal_band_min > terrain_min, but V5
   pins both at 0.2 m, so the locked default could not instantiate. Fixed the
   validator to allow equality (band min == terrain min is the pin).
2. A test used a wrong "invalid input" case (band above terrain min is valid);
   replaced with genuinely invalid cases (band below terrain min; band past
   terrain max).
3. Missing ProjectConfig import in the rewritten test file; import added.

### Known issues / notes

- The test for the bad-triangle rainfall case documents that a 6 h full
  zero-ended triangle (270 mm at 90 mm/h peak) is rejected by construction
  because its implied pulse would not end strictly inside the storm window.
- The three failures above were config-patch and test-patch corrections
  during Phase 0 itself; no locked V7 numerical value changed in any fix.

### Next phase (awaiting explicit "next")

Phase 1: environment generation. Terrain (trend + spectral noise + two
depressions + sea band, coastal pin enforced), land cover classes, hidden
drainage fields from the independent drainage stream, the four locked
weather scenarios, and the road network with bridge decks. Acceptance:
TC1 determinism, TC2a/TC2b structural independence, TC3 road connectivity
and bridge placement, TC13b datum consistency.

### Corrective audit after the Phase 0 commit (recorded, pending commit approval)

Scope: four material issues from a post-commit audit; no locked V7 value
changed; no history rewrite; branch is main tracking origin/main.

- rng.py: removed the false "different stream names can never collide"
  claim; replaced with accurate wording (deterministic derivation, collision
  negligibly unlikely, not impossible). Algorithm unchanged; golden tests
  unchanged.
- config.py: module-level SCENARIOS registry wrapped in MappingProxyType
  (ProjectConfig view unchanged); new focused test proves registry- and
  member-level mutation fails.
- README.md: "pinned dependencies" wording corrected to minimum-version
  constraints; PowerShell environment setup and dependency sections added.
- Environment: .venv created from Python 3.11.9 (py -3.11); minimum
  constraint set installed project-locally; resolved versions recorded in
  README. .venv/ already ignored by Git (verified with git check-ignore).
- Test result after corrections (project venv, Python 3.11.9, pytest 9.2.0):
  35 passed, 0 failed, 0 skipped in 12.51 s, including TC6.
- Modelling decision recorded (user choice, Option A): inland waterbody
  cells are ordinary dynamic surface-water cells that participate in
  storage and the mass ledger. Agreed scope:
  (1) initial depth from the waterbody spill level against its bed;
  (2) lateral exchange with neighbours and overflow onto land allowed;
  (3) storage included in total surface-water storage;
  (4) ledger tracks change relative to the initial state, so initial lake
  volume is never counted as new rainfall input;
  (5) WATER cells retain alpha = 1.0 under the established partition;
  (6) ocean cells remain the only exclusion (boundary reservoir).
  Documentation consolidated in config.py; implementation belongs to
  Phase 2 and no simulator code exists yet.

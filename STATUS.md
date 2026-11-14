# STATUS

## Phase 1: Environment generation - COMPLETE (audit-corrected)

Date: 2026-11-09
Specification: Architecture V7 (approved)

### Audit corrections applied before commit approval (user review round 1)

Scope: four review findings; no locked V7 value changed; no Phase 0
configuration or RNG code touched; environment digest semantics unchanged
(see item 3 for the one seed whose digest necessarily changed).

1. Road lake-crossing guarantee (`roads.py _ensure_lake_crossings`
   rewritten). A crossing column is now chosen only among the lake's OWN
   water columns, candidates ordered deterministically by descending
   water-cell count and then column id; occupied columns are skipped
   explicitly instead of the previous increment-off-the-lake walk; the
   moved line keeps its service class; no extra RNG draw is consumed; the
   helper raises RuntimeError explicitly if no valid interior crossing
   column exists rather than silently violating the invariant. Unit tests
   cover actual-water-column targets, class preservation and uniqueness,
   the occupied-water-column (crossed, no move) case, an out-of-bounds
   densest-column fallback, two lakes in one call, a move never removing
   another lake's only crossing (seed-78 equivalent), an explicit failure
   when every line is another lake's only crossing, the no-interior-column
   failure path, and determinism: nine crossing-guarantee helper tests.
   New environment-level tests assert connectivity and at least one bridge
   per lake at the master seed, seed 78 (regression), and all five
   Experiment A seeds.
2. Read-only arrays. New `src/urbanflood/arrays.py` with `freeze_array`
   (disables the writeable flag in place, no copy). Applied at final
   assembly to every exposed array of SyntheticEnvironment (8 arrays),
   RoadNetwork (10 arrays), and DrainageFields (3 arrays). Tests assert
   read-only status for every exposed array and that mutation attempts
   raise ValueError; digest computation and all existing consumers were
   confirmed to work on the frozen buffers (TC1 rebuild equality still
   passes). Environment digests for all six tested seeds are byte-identical
   to before the freeze (metadata-only change).
3. Lake carving rim coverage (necessary correction, explicitly explained).
   The recorded spill level was previously the minimum over the rim ring
   [2.05*sigma, 2.35*sigma], although the carve bump's compact support
   already ends at 2.0*sigma. The unchecked annulus [2.0, 2.05]*sigma could
   dip below the recorded spill, so surrounding land would read as
   submerged at a waterbody whose rim should bound it. The spill ring is
   now the FULL non-carve annulus [2.0, 2.35]*sigma (environment docstring
   updated). Guard also added: the complete bowl footprint must lie inside
   the grid columns so no lake is laterally clipped. Effects measured at
   the master seed and the five Experiment A seeds: five digests are
   unchanged; at seed 6574970189797259754 lake 1's spill level dropped
   from 5.8328 m to 5.6363 m (the widened ring found the true rim minimum)
   and its area changed from 36 to 35 cells, so that environment's digest
   changed. Draw order is untouched; the change only re-evaluates the
   spill ring over a wider (still untouched) annulus. The digest comparison
   above is HISTORICAL: it was measured when the rim-ring correction was
   applied (only seed 6574970189797259754's digest changed; the other five
   seeds were byte-identical). Current-state behavior is verified by the
   executed test suite -- TC1 determinism and the six-seed lake invariants
   run on every invocation -- not by digest comparisons. This session's
   lower-row guard correction (2.0*sigma -> 2.35*sigma) changes no
   generated environment: legal placements always satisfy center_row -
   2.35*sigma >= 13.5 (LAKE_ROW_MIN = 30, sigma <= 7), so the tightened
   guard cannot bind.
4. Rainfall volume helper semantics (`tests/test_weather.py` fixed). The
   helper `applied_precipitation_volume` is per cell per timestep; the
   faulty test had passed an aggregate sum of rates. Now parametrized
   tests check it with individual timestep rates (V_precip = R_k * A * dt)
   and a separate, clearly named aggregate helper
   `total_precipitation_volume(rates, cell_area_m2, dt_s)` is provided and
   tested against the stepwise sum. Rainfall totals, pulse timings, and
   scenario values are untouched.

Multi-seed lake evidence (master 2027 plus all five Experiment A seeds;
test, not a universal claim). Actual values:

  seed 2027:                lake 0 spill 10.0624, bed 4.9346, area 19
                            lake 1 spill 11.4668, bed 2.9670, area 12
  seed 10406733617104774339: lake 0 spill 16.3729, bed 11.8350, area 12
                            lake 1 spill 3.4379, bed 0.2000, area 34
  seed 2112024614403005981: lake 0 spill 9.3471, bed 0.2000, area 53
                            lake 1 spill 3.9766, bed 0.2000, area 22
  seed 4656971133199668420: lake 0 spill 12.2401, bed 0.7870, area 45
                            lake 1 spill 8.3411, bed 0.2000, area 44
  seed 11861282352235076592: lake 0 spill 3.0330, bed 0.2000, area 15
                            lake 1 spill 3.4053, bed 0.2000, area 13
  seed 6574970189797259754: lake 0 spill 18.2282, bed 4.6992, area 73
                            lake 1 spill 5.6363, bed 0.2000, area 35

The 14 m adaptive-depth cap binds (carved depth exactly 14.0 m) for every
accepted attempt at all six seeds; acceptance invariants (spill >= 2.0 m,
spill minus bed >= 0.3 m, single 4-connected component, at least 12 cells,
lake cells strictly below spill, entire full-annulus rim at or above the
spill level) hold everywhere, so the binding cap does not hurt validity.
No accepted attempt was rejected by the new footprint guard at any of the
six seeds (five digests identical, one changed only through item 3's rim
fix). Per-lake maximum initial depths h_init = spill - bed range from 2.83 m to
13.53 m across the twelve lakes (h_init itself is evaluated per cell as
max(0, L_normal - z) in Phase 2); several lakes clip to the 0.2 m terrain
floor and are therefore flat-bedded basins, disclosed as such.

### Completed (original Phase 1 entry)

- `src/urbanflood/environment.py`: SYNTHETIC environment builder on the
  independent "env" stream with a documented canonical RNG draw order
  (spectral white noise -> lake placement attempts -> road line permutations
  -> land-cover uniforms):
  - FFT-based spectral relief field (power-law amplitude mask, wavelengths
    below 150 m removed, k = 0 mode zeroed), regional inland trend
    (0.2 -> 52 m, power 1.35, smooth shore approach), amplitude ramping so
    low-lying areas stay flat;
  - vertical datum MSL = 0.0 m enforced: land terrain within [0.2, 60.0] m,
    coastal band rows (4 rows above the shoreline) pinned exactly to
    linspace(1.2, 0.2, 4) (TC13b), ocean band of 8 rows with synthetic
    bathymetry from -1.5 m to -7.5 m below MSL;
  - two carved lakes with compact-support Gaussian bowls cut to guarantee a
    true rim spill level: the spill level is the minimum rim-ring elevation
    of the post-carve terrain, recorded per lake for the Option A initial
    depth rule h_init = max(0, L_normal - z) in Phase 2; effective carve
    depth is adaptive (deterministic function of the draw and terrain,
    capped at 14 m) because lateral relief variation across a bowl routine-
    ly exceeds any fixed small depth; lakes are single 4-connected
    components, inland-placed, separated by at least 30 cells;
  - land cover from road proximity (distance transform) and low-lying
    terrain with the four locked LandCover classes; WATER exactly equals
    ocean plus lake cells;
  - outfalls: land cells with a 4-neighbour face on the ocean; invert at
    the cell bed elevation (freeboard locked at 0.0 m); all 96 shoreline
    cells are outfalls;
  - SHA-256 environment digest over the canonical array sequence (TC1).
- `src/urbanflood/roads.py`: connected SYNTHETIC grid road network
  (arterial/secondary/local lines; locked speeds 50/30/20 km/h):
  - fixed coastal arterial on the shoreline row plus RNG-selected
    horizontal (2 arterial, 4 secondary, 6 local) and vertical (3 arterial,
    4 secondary, 6 local) full-span lines; connectivity holds by
    construction and is asserted on the built graph (TC3, single component,
    2213 nodes / 2355 edges at master seed 2027);
  - bridges: water runs on any line are replaced by single bridge edges
    whose deck elevation follows the locked rule deck =
    max(max crossed lake spill level + 1.0 m freeboard,
    min(abutment land elevations)); each lake is guaranteed at least one
    crossing by a deterministic no-RNG adjustment (move the vertical line
    closest to the densest water column onto that column);
  - canonical node ids (row-major sorted land road cells) and edge ids
    (sorted unordered uv pairs); paths record full geometry cells for the
    later 3-point edge depth sampling;
  - no road path ever touches an ocean cell.
- `src/urbanflood/drainage.py`: hidden drainage fields from the independent
  drainage stream; signature (master_seed, n_cells, blocked_fraction,
  drainage_config) receives NO terrain argument (TC2a structural);
  canonical draw order capacity -> efficiency -> blocked makes the locked
  S3 rule exact by construction: regenerating the mask with
  blocked_fraction = 0.40 on the same stream reproduces capacity and
  efficiency bitwise and yields a nested (superset) blocked mask.
- `src/urbanflood/weather.py`: the four locked scenarios realized as exact
  interval integration of the triangular pulse, R_k = (1/dt) * integral of
  R over [t_k, t_k + dt), verified to preserve the declared totals exactly
  at every tested dt (7.5 to 300 s); midpoint tide series generator
  t_mid,k = (k + 0.5) * dt feeding Phase 2 outfall blocking and sea
  exchange; the weather stream is deliberately left unused (locked
  scenarios are deterministic; zero RNG consumption asserted structurally).
- Tests: tests/test_environment.py, tests/test_roads.py,
  tests/test_drainage.py, tests/test_weather.py.

### Test results (actually executed, final run after audit corrections)

Command: `.\.venv\Scripts\python.exe -m pytest -v`
Result: 134 passed, 0 failed, 0 skipped in 1.26 s
Platform: win32, Python 3.11.9 (project .venv), pytest 9.2.0

Breakdown (per-module counts gathered from individual runs):
- tests/test_ascii.py: 1 passed (TC6 scan, 21 files including arrays.py; all ASCII)
- tests/test_config.py: 25 passed (unchanged Phase 0 behaviour)
- tests/test_drainage.py: 12 passed (TC2a signature, ranges, determinism,
  S3 mask nesting, frozen container, read-only arrays, digest reads)
- tests/test_environment.py: 31 passed (TC1 determinism, TC13b checks,
  lake Option A invariants including the six-seed sweep, land cover,
  outfalls, stream membership, read-only arrays and mutation rejections)
- tests/test_rng.py: 9 passed (unchanged Phase 0 behaviour)
- tests/test_roads.py: 31 passed (TC3 connectivity, canonical ids, bridge
  paths and locked deck rule, six-seed connectivity plus bridges, nine
  crossing-guarantee helper tests)
- tests/test_weather.py: 25 passed (exact integration totals at many dt,
  pulse shape and support, R/S1 totals, S2/S3 inheritance, per-timestep
  and aggregate volume semantics, zero-RNG claim, midpoint tide)

Pre-correction runs: first run 12 failed / 87 passed (three causes
documented below); post-implementation run 100 passed; first audit run
3 failed / 128 passed (two unit-test bugs using unique lake-id values
instead of unique water columns, plus the genuine rim-annulus finding
corrected in item 3). All runs' outputs were captured.

First-implementation run failure causes (all fixed; no locked value changed):

1. weather.py interval_mean_rate iterated the GLOBAL pulse knot list of
   each call instead of the knots clipped to the query interval, so every
   interval received the full-storm integral (applied depth 8.64 m instead
   of 0.12 m). Fixed to intersect the interval with the pulse knots; the
   fine-trapezoid cross-check now passes to 1e-6 relative.
2. A wrong constant in a test (ocean row count written as SEA_ROWS - 1
   instead of SEA_ROWS; the sea band is 8 ocean rows exactly).
3. A test was accidentally parametrized as a fixture (CONFIG in the
   signature of test_free_flow_times_locked_speeds); replaced by the
   module constant.

### Generation hyperparameters (Phase 1 design choices, not locked V7 values)

Recorded in the module constants; changing any of them intentionally
changes the environment and its digest:
sea band 8 rows; coastal band 4 pinned rows; relief amplitude 6.0 m with
8.0 m ramp; spectral exponent 1.6, soft wavelength 1600 m, cutoff 150 m;
trend max 52.0 m at power 1.35; two lakes with sigma 4-7 cells, depth draw
3-6 m (minimum; adaptive cap 14.0 m), spill threshold 2.0 m, minimum area
12 cells, minimum water depth 0.3 m, 40 attempts, center separation
30 cells; road lines 2+4+6 horizontal (plus the fixed coastal arterial)
and 3+4+6 vertical; impervious probability base 0.30, road term 0.30
(length scale 2.5 cells), lowland term 0.15, open-ground fraction 0.15.

### Known issues / notes

- The adaptive lake depth cap (14 m) can in principle bind before
  submergence for an extreme draw; such an attempt is then rejected and
  redrawn deterministically (40 attempts per lake); no failure was
  observed at the master seed or the 5 Experiment A seeds.
- Land cover under lakes is WATER by construction; land-cover impervious
  probability near bridges uses road-node proximity, so abutment cells are
  near-road but bridge water cells are not classified impervious.
- Ocean cells are the only storage exclusion (boundary reservoir); lake
  initial volume belongs to the ledger baseline (Option A). Recorded spill
  levels are consumed (a) by roads.py for bridge deck elevation and (b) by
  src/urbanflood/initial_state.py build_initial_depth() for Option A initial
  depth (added in Stage 10b).

### Correction note: Option A initialisation

Option A lake initialisation (h_init = max(0, spill_level - bed)) was documented
in Phase 1 (environment.py and config.py docstrings) and listed as a Phase 2
acceptance item, but no builder existed until Stage 10b. The builder
src/urbanflood/initial_state.py build_initial_depth() was added afterwards,
and all end-to-end tests (six-hour test, TC12, node-depths, edge-depths,
stage15_diagnostic) were switched from np.zeros initial depth to
build_initial_depth(env).

### Phase 2: Flood simulator - COMPLETE

Date: 2026-11-10

The single-timestep boundary was extended by Stages 9 to 11 (node, edge and
peak outputs).
Specification: Architecture V7 (approved)

### Acceptance record

Phase 2 Milestone 2 (single-timestep flood simulator) accepted at the
single-timestep implementation boundary. The synchronous per-step update
implements:

- Rainfall partition V_runoff/V_loss (V_precip = R_k * A * dt;
  V_runoff = alpha * V_precip enters storage; V_loss never enters storage)
- Drainage with bounded-availability rescaling (capacity * efficiency,
  bounded by donor budget)
- Per-face bidirectional open coastal sea exchange (Option B):
  V_sea_i = c * (eta_k - W_i) * dx * dt; eta_k > W_i inflow (external,
  no donor budget), eta_k < W_i outflow (donor-limited), equal heads
  zero exchange; Z_out,i = cell bed elevation does not gate exchange
- Inter-cell linear head-difference exchange in canonical pair order
- Mass ledger IN - OUT - dS asserted to 1e-9 relative
- Ocean boundary reservoir (excluded from storage)
- Option A lake initialisation from recorded spill levels

TC7 (mass ledger), TC8 (toy problems), and TC14 (bitwise canonical-order
invariance) pass at the single-timestep level. TC14 H_peak permutation
invariance is also verified at the multi-timestep orchestration level
(via run_simulation() with permuted face generation order). TC11 (face-count
proportionality) verified at single-timestep and six-hour levels:
dt=30s yields 12,032,640 total evaluations (16,712 per step x 720 steps);
dt=15s yields 24,065,280 total evaluations (16,712 per step x 1440 steps);
ratio exactly 2.0. TC10 dt sensitivity (dt = 15 s) and TC12 rainfall
bookkeeping have been implemented and verified at multi-timestep level.

### Test results (actually executed)

Quick tests (freshly executed in Stage 8):
- py_compile: successful on tests/test_orchestrator.py
- tests/test_timestep.py: 19 passed (TC7, TC8, TC11 single-timestep, TC14 single-timestep)
- New simulation-level TC14 H_peak permutation invariance: 1 passed, 43 deselected
- Existing single-timestep TC14: 1 passed, 18 deselected
- Quick orchestrator regression (excluding six-hour, TC12 rainfall, TC11 face-count): 41 passed, 3 deselected

Six-hour tests (previously executed in Stages 5-6, historical):
- test_orchestrator_six_hour_execution: passed
- test_orchestrator_tc12_rainfall_normalization: passed (dt=30s and dt=15s, zero error)
- test_orchestrator_tc11_face_count_proportionality: passed (ratio exactly 2.0)

### Stage 9: Node-depth outputs - IMPLEMENTED (freshly executed)

Node-depth output contract (docs/PHASE2_NUMERICAL_SPEC.md section 14):

- What: final water depth in metres at each road-network node cell at the end
  of the simulation. Final depths only (not a time series); peak node depths
  are not part of the required interface (optional and undecided).
- What is a node: a land cell on any road line, given by
  env.road_network.node_rows / node_cols. Lake and ocean cells are never nodes
  (roads.py), so every node cell is a simulated surface-water cell with a
  defined final depth. No ocean/lake special-casing is required beyond correct
  cell indexing.
- Identity and ordering: node id k indexes node_rows[k] / node_cols[k], the
  row-major sorted node list from roads.py. Ordering is canonical and
  deterministic, derived only from grid geometry. Output order follows node id
  order.
- Derivation: depth[k] = result.h_final[node_rows[k], node_cols[k]]. Values are
  taken from h_final (final cell depths), not H_peak. result.h_final is not
  modified; the returned array is a fresh copy (no aliasing).
- Shape and dtype: (n_nodes,), metres. The output dtype follows result.h_final
  exactly; no independent dtype conversion or forced float64 is applied. The
  broader output container/dtype decision remains deferred (spec section 17).
- Scope: implemented as orchestrator.node_depths(result, env). Does not change
  rainfall, tide, routing, face enumeration, timestep logic, mass accounting,
  or edge-depth outputs. No new physical assumptions or interpolation rules.

Implementation: new function node_depths() in src/urbanflood/orchestrator.py.

Test results (freshly executed in Stage 9):
- New focused node-depth tests in tests/test_orchestrator.py: 5 passed
  (contract on real simulation, final-depth-not-peak, node-id ordering,
  mismatched-shape rejection, dtype preservation for float32 source).
- tests/test_timestep.py: 19 passed (unchanged).
- tests/test_numerics.py: 48 passed (unchanged).
- test_orchestrator.py fast suite (excluding six-hour, tc12_rainfall,
  tc11_face_count): 46 passed, 3 deselected (41 historical fast tests +
  5 new node-depth tests).
- test_orchestrator_six_hour_execution: 1 passed in 770.56 s
  (TC7 mass balance < 1e-9; TC10 criteria pass). Confirms existing numerical
  behaviour is unchanged by the additive node_depths function.

Historical results (kept separate, from Stages 5-8):
- test_orchestrator_six_hour_execution: passed (Stage 8)
- test_orchestrator_tc12_rainfall_normalization: passed (dt=30s and dt=15s,
  zero error) (Stage 8)
- test_orchestrator_tc11_face_count_proportionality: passed (ratio exactly
  2.0) (Stage 8)
- Quick orchestrator regression: 41 passed, 3 deselected (Stage 8)
- tests/test_timestep.py: 19 passed (Stage 8)
- Single-timestep TC14: 1 passed, 18 deselected (Stage 8)
- Simulation-level TC14 H_peak permutation invariance: 1 passed, 43 deselected
  (Stage 8)

### Stage 10: Edge-depth outputs (final) - IMPLEMENTED

- edge_depths(result, env) in src/urbanflood/orchestrator.py returns final
  water-depth samples at three positions along each road-network edge, shape
  (n_edges, 3), metres, start/mid/end rule from docs/PHASE2_NUMERICAL_SPEC.md
  section 14. tests: the edge_depths tests in tests/test_orchestrator.py
  (selected by -k edge_depths) plus final-depth fresh-copy tests in
  tests/test_peak_outputs.py.

### Stage 10b: Option A initial-depth builder - IMPLEMENTED

- src/urbanflood/initial_state.py build_initial_depth(env) returns a fresh
  float64 array of shape (n_rows, n_cols): max(0, spill_level[lake_id] -
  elevation_m) on is_lake cells, 0.0 elsewhere. Vectorized, no mutation of env,
  environment digest unchanged.
- Tests: tests/test_initial_state.py, 8 tests (shape/dtype/finite, lake
  formula exact, at least one positive lake cell, fresh array, determinism,
  short-run integration, layer separation).
- Verified numbers (master seed 2027): lake volume at t=0 is 202175.996 m3
  across 31 lake cells; lake 0 max initial depth 5.127741 m, lake 1 max
  initial depth 8.499748 m.

### Stage 11: Peak node and peak edge depths - IMPLEMENTED

- peak_node_depths(result, env): shape (n_nodes,), metres, sampled from
  result.H_peak at canonical node cells. Peak = running maximum of cell depth
  over the whole simulation including initial depth.
- peak_edge_depths(result, env): shape (n_edges, 3), metres, sampled from
  result.H_peak at start/mid/end using the same piecewise-linear interpolation
  rule as edge_depths. SAMPLED peak (start/mid/end interpolated from cell
  peaks), not the maximum over every path cell.
- Both return fresh copies, dtype follows source, validate grid shape.
- 14 tests in tests/test_peak_outputs.py.
- This is a design addition; the definition of effective edge depth d_e is NOT
  yet defined in RoadConfig and remains a Phase 4 specification item
  (proposed: d_e = max of the three peak samples, not yet approved).

Remaining Phase 2 multi-timestep items: output container/dtype decisions are
deferred to the next milestone (spec section 17). film_threshold_depth_m
remains unused in Phase 2.

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

### Corrective audit after the Phase 0 commit (recorded)

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

### Verified results (Stage 10b and Stage 11)

- Six-hour test (Scenario R, seed 2027, Option A initial state): passed in
  358.56 s on the development machine.
- TC10 (six-hour, Option A initial state): D_max 0.011246 m (limit 0.05 m),
  RMSE 0.001120 m (limit 0.01 m), D_storage 0.000533 (limit 0.01). 7.5 s
  fallback not triggered. S30 = 1389307.274347 m3, S15 =
  1388566.239373 m3.
- Quick suite with six-hour test deselected: 271 passed (before Stage 11
  tests were added).
- Total collected tests: 286 (from pytest --collect-only). Pass/fail counts
  for the post-Stage-11 full suite are not recorded.
- Full quick suite after Stage 11 (six-hour test deselected): 283 passed, 1
  deselected in 696.63 s. Two further fresh-copy tests were added afterwards;
  the total collected is now 286.
- Peak versus final edge depth, Scenario R, seed 2027, Option A initial state,
  six-hour run; effective depth is the max over the 3 samples per edge; 2355
  edges: closure threshold 0.20 m: 269 edges closed using final depth, 440
  using peak. Threshold 0.30 m: 234 using final, 396 using peak. Threshold
  0.45 m: 181 using final, 317 using peak. 427 edges have peak minus final
  greater than 0.05 m. Recorded conclusion: routing must consume peak edge
  depths; final depths understate closures by about 41 percent at the default
  0.30 m threshold (234 versus 396).
- Limitation: peak edge depth is sampled at 3 positions (start, midpoint, end),
  so a flooded stretch between samples can be missed.

### Decisions recorded

1. Drainage applies to lake cells: the spec has no lake exemption, and
   drainage is computed for every non-ocean cell including inland lakes.
   Measured impact: on environment seed 2027, Scenario R, lakes lose about
   2429 m3 (1.2 percent of initial lake volume) over 120 pre-rain steps,
   entirely due to drainage. Disabling drainage on lake cells changed no land
   flood peak (max land H_peak 2.7505 m in both runs) and left the count of
   road nodes with peak depth above 0.30 m unchanged (252 in both). Documented
   as a modelling limitation, not tuned. See README.md Known limitations.
2. Layer boundary: the susceptibility layer (Phase 3) must not import or read
   drainage, outfalls, simulated depths, the simulator, the orchestrator,
   initial_state, or lake spill levels / lake_specs. It may use elevation_m,
   is_ocean, is_lake, land_cover, road_cells only.
3. Routing consumes PEAK edge depths (peak_edge_depths). The definition of
   effective edge depth d_e is NOT yet defined in RoadConfig and is a Phase 4
   specification item. Proposed: d_e = max of the three peak samples (not yet
   approved).

### Open items

- film_threshold_depth_m (0.005 m) unused in Phase 2; no role established.
- Bridge-deck inundation not modelled; edge-depth samples on bridge/lake cells
  represent underlying lake terrain cell depth, not deck depth.
- Phase 4 specification: see docs/PHASE4_ROUTING_SPEC.md.

## Phase 3: Susceptibility index and Experiment A - COMPLETE

Date: 2026-11-13
Specification: Architecture V7 (approved); Phase 3 additions adopted
(author-delegated, Stage 12); docs/PHASE3_SUSCEPTIBILITY_SPEC.md.

### Modules

- `src/urbanflood/susceptibility.py`: build_susceptibility(env, cfg) ->
  SusceptibilityResult; static weighted susceptibility index from terrain,
  land cover, and water masks only; leakage-free (never imports or reads
  drainage, outfalls, simulated depths, H_peak, simulator, orchestrator,
  initial_state, or roads).
- `src/urbanflood/experiment_a.py`: spearman, roc_auc, top_decile_lift,
  ablated_index, permuted_scores, evaluate_methods, summarize; pure
  functions importing only numpy and scipy.stats; no urbanflood imports.
- `run_experiment_a.py`: experiment runner at repository root; builds
  environments, runs simulations, writes results.csv and summary.txt;
  resumable via cached .npy files; refuses --out inside the repository.

### Docs

- `docs/PHASE3_SUSCEPTIBILITY_SPEC.md`: feature scores (elevation, TPI,
  water distance, land cover, slope), index weights, Experiment A
  specification, layer separation test (TC4), and Experiment A results
  (section 9).
- `docs/experiment_a/`: results.csv, summary.txt, README.txt from the
  Experiment A run (5 seeds x 4 scenarios = 20 runs).

### Tests

- tests/test_susceptibility.py: 17 tests (feature scores on toy fixtures,
  weighted sum, range/masks, determinism, TC4 structural and behavioural,
  clip fractions, multi-seed, index statistics).
- tests/test_experiment_a.py: 30 tests (spearman, roc_auc, top_decile_lift,
  ablated_index, permuted_scores, evaluate_methods fields, import
  isolation, summarize wins/losses/ties/skipped/mean-diff, missing seed,
  NaN handling, cross-product vs per-seed pairing, all seven metrics,
  all scenarios present).
- Collected test count for these two files: 47 (from pytest
  --collect-only).

### Pointer

- Full results in docs/experiment_a/.

### Results summary (Experiment A, mean over 5 seeds, land cells)

- auc_030: R index 0.722172 vs elevation_only 0.668987 vs random 0.500241;
  S1 index 0.741037 vs elevation_only 0.700976; S2 index 0.819635 vs
  elevation_only 0.782328; S3 index 0.724680 vs elevation_only 0.672852.
- spearman: R index 0.363357 vs elevation_only 0.270281; S1 index 0.373064
  vs elevation_only 0.278820; S2 index 0.423517 vs elevation_only 0.332318;
  S3 index 0.363685 vs elevation_only 0.270684.
- index vs elevation_only on auc_030 (per-seed paired): wins 5 of 5 in
  each of R, S1, S2, S3; mean differences R 0.053185, S1 0.040060,
  S2 0.037307, S3 0.051828.
- Elevation clip on seed 2027: 19.6 percent of scored cells above the 40 m
  bound score 0 on the elevation feature.
- First end-to-end simulation of scenarios S1, S2, S3 completed (20 runs,
  720 steps, dt 30 s); scenario R had been run earlier.

### Correction note

Statement (b) in docs/PHASE3_SUSCEPTIBILITY_SPEC.md section 9 was corrected:
S3 lift_030 was previously misreported; elevation-only has higher top-decile
lift at 0.30 m in S1, S2, and marginally S3 (3.362287 vs 3.352162), while the
index is higher at 0.30 m only in R (3.341657 vs 3.288309).

## Phase 4: Routing and Experiment C - IN PROGRESS

- `src/urbanflood/routing.py`: effective_edge_depth, sample_od_pairs,
  policy_costs, solve_routes, route_metrics; pure functions importing only
  numpy, networkx, dataclasses, typing, collections.abc, __future__; no
  urbanflood imports; 35 tests (from pytest --collect-only).
- `tests/test_routing.py`: effective_edge_depth, sample_od_pairs,
  policy_costs, solve_routes, route_metrics, P0/policy consistency, layer
  structure, determinism, timing; 35 tests.
- `src/urbanflood/experiment_c.py`: aggregate_policy_rows,
  check_policy_invariants; pure functions importing only numpy,
  dataclasses, typing, collections.abc, __future__; no urbanflood imports.
- `tests/test_experiment_c.py`: aggregate_policy_rows,
  check_policy_invariants, import isolation; 11 tests
  (from pytest --collect-only).
- `run_experiment_c.py`: experiment runner at repository root; runs
  Experiment C using cached Experiment A H_peak arrays; writes pairs.csv,
  edge_summary.csv, summary.csv, summary.txt, run_meta_c.txt; includes
  cache guard (refuses --out inside repo and mismatched steps/dt).
- Experiment C full run: not yet executed (pending).

## Phase 0: Scaffold and configuration - COMPLETE

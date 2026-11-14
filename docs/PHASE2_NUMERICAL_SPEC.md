# Phase 2 Numerical Specification: Flood Simulator

**SYNTHETIC DATA PROJECT.** All parameters, seeds, scenarios, and results are
synthetic. Nothing here represents real-world Mumbai geography, drainage, tides,
or flood events.

Specification: Architecture V7 (approved). Phase 2 numerical decisions recorded
below are newly approved engineering tolerances and equations; they are not
values recovered from the original V7 specification. Where V7 already locks a
value (grid, timesteps, exchange coefficient c, scenarios, drainage fields,
rainfall, tide, mass ledger), this document preserves it verbatim.

---

## 1. Scope

Phase 2 implements the dynamic flood simulator only:

- Water-depth evolution over the six-hour simulation.
- Rainfall, infiltration/loss partition, drainage, neighbour exchange, and sea-boundary exchange.
- Mass-conservation accounting.
- Timestep sensitivity using the prescribed 30-second and 15-second timesteps.
- Required simulator outputs and their tests.

Out of scope for Phase 2: static flood susceptibility, risk-aware routing,
Streamlit dashboard, external datasets or APIs, real-world calibration or
validation, Phase 3+ functionality.

---

## 2. Domain, time, and grid (locked V7)

| Quantity | Value | Source |
|---|---|---|
| Grid dimensions | 96 x 96 cells | `GridConfig` |
| Cell size `dx` | 50 m | `GridConfig` |
| Cell area `A` | 2,500 m^2 | `GridConfig.cell_area_m2` |
| Domain size | 4.8 km | `GridConfig.domain_size_m` |
| Main timestep `dt` | 30 s | `ExchangeConfig.timestep_s` |
| Sensitivity timestep | 15 s | `ExchangeConfig.sensitivity_timestep_s` |
| Simulation duration | 6 h (21,600 s) | `storm_window_s` |
| Main timesteps | 720 | 21,600 / 30 |
| Sensitivity timesteps | 1,440 | 21,600 / 15 |
| Sensitivity timesteps (fallback) | 2,880 | 21,600 / 7.5 |
| Cell ordering | row-major, index = row * n_cols + col | `SyntheticEnvironment` docstring |
| Vertical datum | MSL = 0.0 m | `ElevationConfig.msl_m` |

Main simulation uses synchronous state updates with one old-state snapshot per
timestep.

---

## 3. Rainfall (locked V7; use existing `weather.py` unchanged)

Rainfall is applied by exact interval integration through `weather.hyetograph`:

- `R_k = (1/dt) * integral of R(t) over [t_k, t_k + dt)` in m/s.
- `V_precip_i,k = R_k * A * dt` in m^3 per cell per timestep
  (`weather.applied_precipitation_volume`).

Rainfall is applied to land cells and inland lake cells; ocean cells receive no
rainfall into simulated surface-water storage (ocean cells are excluded from
storage and act as the boundary reservoir).

Partition (locked V7):

- `V_runoff_i,k = alpha_i * V_precip_i,k` enters surface-water storage.
- `V_loss_i,k = V_precip_i,k - V_runoff_i,k` never enters storage.
- `alpha_i` by land cover: impervious 0.90, open_ground 0.50, vegetation 0.30,
  water 1.00 (`RunoffConfig.coefficients`).

Inland lake cells (Option A) have `alpha = 1.0`.

Same-timestep runoff is available through the donor budget (section 7) and may
fund drainage, internal outgoing exchanges, and sea outflows in the same
timestep. All fluxes are still calculated from the old-state snapshot.

---

## 4. Initial state (Option A, locked V7)

Initial depth per cell:

- `h_init,i = max(0, L_normal,i - z_i)` for inland lake cells,
  where `L_normal,i = spec.spill_level_m` from `LakeSpec`.
- `h_init,i = 0` for all non-lake land cells.
- Ocean cells: excluded from surface-water storage (no initial depth array entry
  relevant to the ledger).

Initial storage per cell: `S_init,i = h_init,i * A` (m^3).
Initial total storage: `S_init = sum(S_init,i)` over simulated surface-water
cells (land + inland lake; not ocean).

`h_init` includes the initial lake depth; H_peak tracking includes the initial
depth as the starting maximum (a cell that never floods still reports its initial
depth as its peak).

Road/bridge cells: roads are terrain features embedded in `z`; there is no
separate road storage layer in Phase 2. Road cells participate in exchange and
drainage like any other land cell according to their land-cover class and
elevation. Bridges do not alter surface-water storage in Phase 2.

---

## 5. Cell water-surface elevation and head difference (new Phase 2 decision)

Cell water-surface elevation:

```
W_i = z_i + h_i
```

For each unique internal face between cells i and j:

```
delta_W_ij = W_i - W_j
V_ij = c * delta_W_ij * dx * dt
```

- Signed-flux conversion to directed donor requests:
  - If `V_ij > 0`: donor is `i`, receiver is `j`, proposed volume is `V_ij`.
  - If `V_ij < 0`: donor is `j`, receiver is `i`, proposed volume is `abs(V_ij)`.
  - If `V_ij = 0`: no outgoing request, but the eligible face still counts as
    evaluated for TC11.
- `c = 0.1 m/s` (locked, `ExchangeConfig.exchange_velocity_m_per_s`).
- Each unique internal face evaluated exactly once per timestep.
- All proposed internal exchanges calculated from the old timestep snapshot.

Dimensional check: `(m/s) * m * m * s = m^3`. Correct.

---

## 6. Internal face enumeration (new Phase 2 decision)

Eligible internal faces:

- All 4-connected adjacent cell pairs (horizontal and vertical neighbours)
  within the grid.
- Both cells must be simulated surface-water cells (land or inland lake;
  ocean cells excluded).
- A face is unique: the unordered pair `{i, j}` is enumerated once, not once per
  endpoint. Cell indices are ordered canonically (lower cell index first) to
  form a stable face ID derived only from grid geometry.
- Ocean-facing faces (one cell ocean, one cell land) are excluded from internal
  exchange; sea exchange is handled separately through outfall faces (section 8).

Canonical ordering: faces are sorted by face ID (cell index pair) for
deterministic accumulation and ledger tallies (TC14 bitwise order invariance).

---

## 7. Drainage (locked V7 fields; new Phase 2 volume rule)

Drainage fields come from `drainage.generate_drainage(master_seed, n_cells,
blocked_fraction, drainage_config)` - no terrain input (TC2a structural
independence).

- `C_i` = capacity in mm/hour (`DrainageFields.capacity_mm_per_h`).
- `eta_i` = efficiency fraction (`DrainageFields.efficiency`).
- `b_i` = blocked boolean (`DrainageFields.blocked`).

Potential drainage volume per cell per timestep:

```
V_potential_i = (C_i / (1000 * 3600)) * A * dt * eta_i
```

Dimensional check: `(mm/h / (1000*3600)) = m/s`; `(m/s) * m^2 * s = m^3`. Correct.

- Blocked drainage (`b_i = True`) removes zero water.
- Unblocked drainage is limited by available surface-water volume.
- Drainage applies to every non-ocean cell, including inland lake cells; the
  spec has no lake exemption. See README.md Known limitations for measured
  impact.

Every outgoing request is a directed transfer with exactly one donor cell.
Internal faces are directed from donor to receiver; drainage and sea outflows
have the land cell as donor. Sea inflows are external inputs and do not
consume a land-cell donor budget.

Donor-availability rule (new Phase 2 decision):

For each donor cell `i`:

1. Calculate all proposed outgoing transfers (internal outflows, drainage,
   and sea outflows) from the old state.
2. `Q_i = sum(all proposed internal outflows from i) + V_potential_i (if
   unblocked) + sum(proposed sea outflows from i)`.
3. `budget_i = h_now_i * A + V_runoff_i,k` (m^3).
4. `scale_i = min(1, budget_i / Q_i)` when `Q_i > 0`; otherwise `scale_i = 1`.
5. Every outgoing request from donor `i` receives the same `scale_i`.
6. For an internal face directed from `i` to `j`:
   `V_actual_ij = scale_i * V_proposed_ij`.
   The exact same `V_actual_ij` is subtracted from `i` and added to `j`.
   Independent scaling factors are never applied to the two endpoints of an
   internal face.
7. Apply all scaled transfers synchronously from the old state.
8. Resulting depths remain non-negative without silently clipping away mass
   beyond the availability cap.

Incoming transfers in the same timestep are not counted as available for
outgoing transfers (no order-dependent cascading within a timestep).

---

## 8. Sea-boundary exchange (new Phase 2 decision)

The coastal boundary is an open land-ocean interface, not a protected drainage
outfall with a flap valve. Sea exchange is driven by the head difference between
the sea surface and the land cell water surface.

Outfall faces: land cells with a 4-neighbour face on the ocean
(`SyntheticEnvironment.outfall_mask`), invert at cell bed elevation
(`outfall_invert_m`, freeboard 0.0 m). `Z_out,i` remains defined as the cell bed
elevation but does **not** gate open coastal exchange.

- Each unique land-ocean face has a canonical face ID derived from grid
  geometry.
- Each eligible land-ocean face is evaluated exactly once per timestep. Under the
  configured grid geometry each coastal land cell has exactly one ocean-facing
  edge, so there is exactly one eligible sea face per coastal land cell.
- Sea-face instrumentation is separate from TC11's internal-face count.

Midpoint tide `eta_k = eta(t_k + dt/2)` from `weather.tide_midpoint_series`,
evaluated exactly once per timestep and used consistently for all tide-dependent
calculations (coastal sea exchange).

Cell water-surface elevation:

```
W_i = z_i + h_i
```

Sea exchange per eligible outfall face:

```
V_sea_i = c * (eta_k - W_i) * dx * dt
```

- `eta_k > W_i`: sea inflow into the land cell (positive `V_sea`,
  accumulate in SEA_IN). Sea inflows are external inputs and do not consume a
  land-cell donor budget.
- `eta_k < W_i`: sea outflow from the land cell to the sea (negative `V_sea`,
  accumulate magnitude in SEA_OUT after donor scaling).
- `eta_k = W_i`: zero exchange (`V_sea_i = 0`).

Dimensional check: `(m/s) * m * m * s = m^3`. Correct.

Sea exchange outflows participate in the same donor-availability scaling as
drainage and internal transfers (section 7). Sea exchange inflows are external
inputs and do not participate in donor-availability scaling.

Ocean cells remain excluded from simulated surface-water storage.

Sea-boundary coefficient check:

```
m_i * c * dt / dx
```

where `m_i` is the number of ocean-facing edges for land cell `i`. Under the
configured grid geometry (ocean occupies rows `shore_row + 1` to `n_rows - 1`,
all columns; land cells bordering the ocean are exactly row `shore_row` cells,
each with exactly one south-facing ocean neighbour), `m_i,max = 1`.

| dt (s) | c (m/s) | dx (m) | m_i,max | m_i * c * dt / dx |
|--------|---------|--------|---------|-------------------|
| 30     | 0.1     | 50     | 1       | 0.06              |
| 15     | 0.1     | 50     | 1       | 0.03              |

This coefficient bounds the sea-boundary contribution to the per-cell explicit
update. It is a design check, not a proof of unconditional stability for the
complete coupled simulation (which additionally couples internal exchange,
drainage, and rainfall). Non-negative storage is enforced by the donor-availability
limiter (section 7); full numerical stability is verified empirically (TC8,
TC10, TC12).

---

## 9. Synchronous update (locked V7)

One old-state snapshot `h_now` per timestep feeds ALL candidate quantities
(drainage, per-face sea exchange, inter-cell pair fluxes).

Per-timestep update:

```
h_new,i = h_now,i + (V_runoff_i,k + sum(scaled inflows) - sum(scaled outflows)) / A
```

- All fluxes are computed from `h_now`; `h_new` is written once per cell.
- Same-timestep runoff `V_runoff_i,k` is included in the donor budget and is
  available to fund outgoing transfers in the same timestep.
- Incoming transfers are not available for further outgoing transfers in the
  same timestep (no order-dependent cascading).
- All transfers are applied simultaneously; no sequential cell updates.

---

## 10. Mass ledger (locked V7)

```
IN      = sum(V_precip) + SEA_IN
OUT     = sum(V_loss) + sum(V_drain) + SEA_OUT
dS      = S_final - S_initial
Residual = IN - OUT - dS
```

- `sum(V_precip)` over all simulated surface-water cells and all timesteps.
- `sum(V_loss)` over all cells and all timesteps.
- `sum(V_drain)` = total drained volume (after availability scaling).
- `SEA_IN` = total positive sea flux into land cells (sea inflow when
  `eta_k > W_i`). This term is physically reachable under bidirectional coastal
  exchange; it is not a placeholder.
- `SEA_OUT` = total magnitude of negative sea flux out of land cells
  (sea outflow when `eta_k < W_i`).
- `S_initial` = `sum(h_init,i * A)` over simulated surface-water cells.
- `S_final` = `sum(h_final,i * A)` over simulated surface-water cells.
- Initial inland-lake volume is baseline storage, not new input (Option A).

Acceptance: relative mass-balance error `|Residual| / max(IN, OUT, 1.0)` must
remain below `1e-9` (TC7).

No artificial residual correction or mass hiding through clipping.

---

## 11. Tide (locked V7; use existing `weather.py` unchanged)

- `eta(t) = A_t * sin(2 * pi * t / T)` with `T = 44712 s`, phase = 0, surge = 0.
- Analytic peak at `t = T/4 = 11178 s`.
- `eta_k = eta(t_k + dt/2)`, exactly one evaluation per timestep
  (`weather.tide_midpoint_series`).
- Same `eta_k` used for coastal sea exchange in a timestep.
- Scenarios: R/S1/S3 `A_t = 0.8 m`; S2 `A_t = 1.8 m`.

---

## 12. Test contracts

### TC7 - Mass conservation

Verify the complete simulator mass ledger across a meaningful six-hour
simulation (not a trivial zero-flow case). Relative mass-balance error below
`1e-9`.

### TC7b - Hand-checkable rainfall partition

Controlled fixture:
- `alpha = 0.90`.
- `V_precip = 100 m^3`.
- `V_runoff = 90 m^3`.
- `V_loss = 10 m^3`.
- Storage increase = 90 m^3.
- Ledger residual = exactly zero.

### TC8 - Physical and directional behaviour

Controlled, deterministic fixtures exercising the actual simulator:

- **Non-negative depth:** after the complete update at every timestep, no cell
  has negative water depth.
- **Bowl pooling:** in a bowl-shaped synthetic terrain fixture, water pools in
  the bowl interior and does not drain below the terrain floor.
- **Blocked versus unblocked drainage:** in an otherwise identical fixture,
  blocked drainage produces greater/deeper water accumulation than unblocked
  drainage.
- **Tide isolation:** compare higher versus lower coastal tide under otherwise
  identical conditions, where the relevant outfall faces exchange freely
  during positive sea-flux intervals so that tide amplitude is the isolated
  variable. Do not assume greater tide amplitude always produces greater
  coastal accumulation if the head-difference relationship between cells
  changes between runs.
- No arbitrary expected numerical outputs are asserted without a
  hand-calculable basis.

### TC11 - Face-count proportionality

Exchange evaluation count = (number of unique eligible internal faces evaluated
per timestep) * (number of simulation timesteps).

For the same six-hour duration and unchanged eligible-face mask:
- dt=30s: 720 timesteps.
- dt=15s: 1,440 timesteps.
- The 15s evaluation count must be exactly twice the 30s count.

Sea faces are excluded from this count (they are counted in separate
instrumentation). Test the actual face enumeration and simulator
instrumentation, not just a hardcoded arithmetic ratio.

### TC12 - Rainfall normalization

Use the existing interval-integrated rainfall implementation unchanged.

For the complete six-hour simulation over rainfall-receiving cells:

```
P_sim = total_precipitation_volume / total_rainfall_receiving_area
```

Compare `P_sim` with the analytical declared storm total in metres.
Absolute depth error <= `1e-12 m`. Run independently at dt=30s and dt=15s.

Do not include ocean cells in the rainfall-receiving area if the simulator does
not apply rainfall to them.

### TC13a - Tide

- Verify every simulator midpoint tide value against the analytic tide function
  evaluated at the corresponding midpoint.
- Verify the continuous analytic maximum separately (`A_t` at `t = T/4`).
- Do not require the maximum discrete midpoint sample to equal the continuous
  analytic maximum exactly.

### TC14 - Bitwise canonical-order invariance

- Candidate exchange generation order can be deliberately permuted in the test.
- Candidates are canonicalised before accumulation (stable face IDs derived only
  from grid geometry, sorted canonically).
- Per-cell floating-point accumulation follows a deterministic order.
- Ledger reductions follow a deterministic order.
- No unordered iteration or nondeterministic parallel reduction affects results.
- Internal face transfers remain equal and opposite (the exact same
  `V_actual_ij` is subtracted from the donor and added to the receiver).

Require bitwise equality for all single-timestep outputs exposed by StepResult:
- Final depth array (`h_new`).
- Total precipitation.
- Total loss.
- Total drainage.
- SEA_IN.
- SEA_OUT.
- Final storage.
- Mass residual.

`H_peak` array is a simulation-level accumulated output (peak water depth per cell
over the entire simulation, including initial depth as its starting maximum). It
is not part of the single-timestep `StepResult` and is not compared by the
single-timestep TC14 permutation test. `H_peak` is required output for the future
multi-timestep simulation/orchestration layer and is deferred until that layer
exists; its permutation invariance must be verified there.

Use actual array byte equality or equivalent bitwise comparison, not
approximate numerical comparison.

---

## 13. V3 timestep sensitivity (new Phase 2 engineering tolerances)

Compare the six-hour simulation at dt=30s against dt=15s using identical
environments, initial conditions, scenario, rainfall, drainage fields, blocked
masks, tide parameters, and physical parameters.

Metrics, calculated across all simulated surface-water cells (excluding ocean
cells), including dry cells and inland lakes:

```
D_max = max(abs(H_peak_30 - H_peak_15))
RMSE  = sqrt(mean((H_peak_30 - H_peak_15)^2))
```

Final storage relative difference:

```
D_storage = abs(S30 - S15) / max(abs(S30), abs(S15))
```

If both `S30` and `S15` are zero, `D_storage` is defined as zero.

| Metric | Tolerance |
|---|---|
| Maximum absolute cell-wise difference in H_peak (`D_max`) | <= 0.05 m |
| H_peak RMSE | <= 0.01 m |
| Relative difference in final total surface-water storage (`D_storage`) | <= 1% |

Apply the same definitions to the 15s versus 7.5s fallback comparison:
if any 30s-versus-15s criterion fails, run dt=7.5s and compare 15s versus 7.5s
using the same metrics and tolerances. If that also fails, report
non-convergence. Do not keep halving indefinitely, do not modify physical
parameters, do not weaken tolerances.

Do not interpolate outputs. These are engineering acceptance criteria, not a
mathematical proof of convergence.

---

## 14. Simulator outputs

Required outputs for later phases:

- `H_peak`: peak water depth per grid cell over the simulation, shape
  `(n_rows, n_cols)`, metres. Includes initial depth as starting maximum.
  Simulation-level accumulated output; not part of the single-timestep
  `StepResult`. Required for the future multi-timestep simulation/orchestration
  layer; deferred until that layer exists.
- Node depths: **final** water depth at road-network node cells, shape
  `(n_nodes,)`, metres, indexed by node id.
- Edge depth samples: **final** water depth at three positions along each
  road-network edge, shape `(n_edges, 3)`, metres, indexed by edge id.
  Column 0 is the canonical start (0% of edge length), column 1 is the
  geometric midpoint (50% of edge length), and column 2 is the canonical end
  (100% of edge length).

  Canonical edge orientation follows the canonical endpoints: the start
  corresponds to the node referenced by `edge_u` and the end corresponds to
  the node referenced by `edge_v`. The `edge_path_cells` sequence for every
  edge begins at the `edge_u` endpoint cell and ends at the `edge_v` endpoint
  cell, as guaranteed by the road generator's increasing-coordinate line
  traversal (`src/urbanflood/roads.py`, `_line_edges` and `build_road_network`).

  Sample depths are computed by one-dimensional piecewise-linear
  interpolation along the ordered `edge_path_cells` sequence. For a path
  containing N cells with depths h_0, ..., h_{N-1} taken from
  `result.h_final`, a fractional edge position f in {0, 0.5, 1} maps to the
  path coordinate x = f * (N - 1). Let i = floor(x) and alpha = x - i. The
  sample is h(f) = (1 - alpha) * h_i + alpha * h_{i+1} when x is not an
  integer, and h(f) = h_i when x is an integer. Consecutive path cells are
  equally spaced at one cell size (50 m), so fractional positions correspond
  to equal fractions of edge length.

  Ordinary edges contain two adjacent cells; their three samples are therefore
  the first cell depth, the arithmetic mean of the two adjacent cell depths
  (midpoint), and the second cell depth. This midpoint is a one-dimensional
  interpolation along the road edge, not an average of independent node-depth
  outputs. Bridge edges use their complete ordered path, including both land
  abutments and intervening water cells, with the same three fractional
  positions regardless of path length. For example, a three-cell path samples
  indices 0, 1, 2; a four-cell path samples indices 0, 1.5, 3 with
  interpolation at the midpoint.

  Interpolated midpoint values are estimates along the one-dimensional edge
  path and may not correspond to actual grid-cell depths. No bilinear
  interpolation or off-path neighbouring-cell interpolation is used. This is
  an output-extraction convention and does not modify the numerical
  simulation. All path cells are non-ocean cells with defined final depths in
  `result.h_final`; ocean cells never appear in edge paths.

  Interpretation note: edge-depth samples are terrain-cell water-depth
  estimates extracted along the road-network edge path. For bridge paths,
  samples falling on lake cells represent water depth in the underlying lake
  cells, not water depth on the bridge deck. The current model does not
  simulate bridge-deck elevation, bridge clearance, or bridge-deck inundation.
  These samples must not be used to claim that a bridge is passable or
  impassable based solely on their values.

Node and edge required outputs are **final depths at the end of the
simulation**. Peak node depths and peak edge depths were added in Stage 11
as `peak_node_depths()` and `peak_edge_depths()`, sampled from `H_peak`
using the same start/mid/end rule as `edge_depths()`. They are SAMPLED peaks
(start/mid/end interpolated from cell peaks), not the maximum over every cell
on the path. H_peak provides peak grid-cell depth; peak node depths are
derivable as H_peak indexed by road-node cells; peak edge depths are sampled
from H_peak along each edge path.

Ownership: `SimulationResult.h_final` and `SimulationResult.H_peak` are stored
fields of the result dataclass; callers receive a reference to the stored
array, and mutating the returned array modifies the SimulationResult field.
The `node_depths()` and `edge_depths()` functions return fresh copies; caller
mutation of their return values does not affect `h_final` or any internal
simulation state.

Proposed implementation decision (not yet implemented): outputs should be
returned as a documented dataclass containing NumPy arrays. Exact array dtypes
should be specified during implementation. This is a proposed decision, not an
already-implemented interface.

Units are metres for all depth outputs. The simulator is independent of the
future susceptibility and routing implementations; outputs are defined with
clean, documented interfaces.

Road/edge ordering: edge ids follow canonical ordering by unordered cell pair
(`src/urbanflood/roads.py` lines 209-240). Edge `edge_path_cells` stores the
full path cell sequence per edge, not precomputed samples. Edge path
orientation begins at the `edge_u` endpoint cell and ends at the `edge_v`
endpoint cell. Sample column ordering is start (column 0), midpoint (column
1), end (column 2).

---

## 15. film_threshold_depth_m

`ExchangeConfig.film_threshold_depth_m = 0.005` is defined in the locked
configuration (`src/urbanflood/config.py:399`) but no role or activation
threshold is established for it in the existing architecture or this document.
It is documented as unused in Phase 2 pending an explicit decision. No new
physical role or activation threshold is invented for it.

---

## 16. Reference scenario

Scenario R is recorded as the proposed default reference scenario for baseline
six-hour simulation and V3 sensitivity. Scenario selection is configurable and
R is not hardcoded throughout the simulator.

---

## 17. Remaining unresolved decisions

The following items are not established by the existing architecture or the
authorized decisions above. They are listed here rather than silently chosen:

1. **H_peak (cell-level peak water depth over the simulation):** required
   simulation-level output, deferred until multi-timestep orchestration exists.
   Includes initial depth as its starting maximum. Not part of the single-timestep
   `StepResult`; must be verified by the future multi-timestep integration test.
   Peak node depths and peak edge depths were added in Stage 11 as sampled
   peaks from H_peak (peak_node_depths, peak_edge_depths). They use the same
   start/mid/end rule as edge_depths applied to H_peak. Peak node depths are
   derivable as H_peak indexed by road-node cells; peak edge depths are
   SAMPLED peaks (start/mid/end interpolated from cell peaks), not the maximum
   over every path cell.
2. **Output container and array dtypes:** Outputs are returned as NumPy arrays
   from standalone functions (`node_depths`, `edge_depths`). Array dtype follows
   `result.h_final` (float64 in the current implementation); no independent
   dtype conversion is applied. A documented dataclass container may be
   considered in a future phase but is not implemented here.

These will be resolved at implementation review time.

---

## 18. Consistency with locked V7

Verified against existing source (no source modified by this document):

| Decision | Locked in V7? | Location |
|---|---|---|
| Grid 96x96, dx=50m, A=2500m^2 | Yes | `config.py:103-123` |
| dt=30s, sensitivity=15s | Yes | `config.py:397-398` |
| 6-hour duration | Yes | `config.py:220` |
| c=0.1 m/s | Yes | `config.py:396` |
| `f_step = 2*c*dt/dx <= 0.20` | Yes | `config.py:414-416` |
| Rainfall partition V_runoff/V_loss | Yes | `config.py:19-30` |
| alpha coefficients | Yes | `config.py:179-182` |
| Drainage fields (no terrain input) | Yes | `drainage.py:67-72` |
| Outfall at cell bed, freeboard 0.0 | Yes | `config.py:145,358-360` |
| Option A lake initialisation | Yes | `config.py:36-37`, `environment.py:36-37` |
| Mass ledger IN/OUT/dS | Yes | `config.py:27-30`, `README.md:58-59` |
| Synchronous update | Yes | `config.py:49-54`, `README.md:61-64` |
| Midpoint tide, one eval per step | Yes | `config.py:261-262`, `weather.py:119-131` |
| Tide T=44712s, phase=0, surge=0 | Yes | `config.py:266-268` |
| Mass ledger relative tolerance 1e-9 | Yes | `config.py:30`, `README.md:59` |
| Coastal boundary: open land-ocean interface, bidirectional exchange | No - Phase 2 decision | `config.py:145,358-360`, `environment.py:308-314` |
| `Z_out,i` (cell bed invert) defined but not gating sea exchange | No - Phase 2 decision | `config.py:145,358-360` |

New Phase 2 engineering decisions (not in original V7): `delta_W` definition and
sign convention, drainage volume equation, donor-availability scaling rule, sea
exchange equation, TC11/TC12 exact constructions, TC13a discrete-vs-continuous
handling, V3 timestep-sensitivity tolerances and fallback, output definitions,
film_threshold_depth_m unused in Phase 2.

---

**End of Phase 2 numerical specification.**

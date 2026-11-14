# Phase 3 Susceptibility Specification

**SYNTHETIC DATA PROJECT.** All parameters, fields, and results are synthetic.
Nothing here represents real-world Mumbai geography, drainage, tides, or flood
events. This index is a static, leakage-free modelling construct; it is not a
flood prediction model and makes no claim about real flood behaviour.

Specification: Architecture V7 (approved). Phase 3 additions recorded below are
new engineering decisions; where V7 already locks a value (grid, cell size,
weights, normalization ranges), this document preserves it verbatim.

---

## 1. Purpose

The susceptibility index is a static weighted score computed from terrain, land
cover, and water masks only. It is leakage-free: it never sees drainage fields,
outfall data, simulated depths, or any simulator output. It is used in Phase 3
(Experiment A) to compare static susceptibility against simulated peak depths
after the fact, as synthetic recovery testing only.

---

## 2. Inputs allowed and forbidden

Allowed inputs (read from SyntheticEnvironment and GridConfig only):

- elevation_m: terrain elevation in metres above MSL
- is_ocean: boolean ocean mask
- is_lake: boolean inland lake mask
- land_cover: integer land cover class codes
- cell_size_m: grid cell size in metres (from GridConfig)

Forbidden inputs (must not be imported, read, or referenced):

- drainage capacity, efficiency, or blocked fields
- outfall_mask, outfall_invert_m
- lake_specs / spill levels
- simulated depths (h_final, h_initial)
- H_peak
- initial_state module
- simulator or orchestrator modules
- roads module

---

## 3. Feature scores (each in [0,1], 1 = more susceptible)

All feature scores are computed on the feature elevation field (see section 4).
Ocean cells receive NaN in every feature array.

### 3.1 Elevation score

score = 1 - clip(z / 40.0, 0, 1)

Uses SusceptibilityConfig.elevation_range_m = (low, high) = (0.0, 40.0).
Equivalently: score = (high - clip(z, low, high)) / (high - low).

Clip fractions are reported: the share of scored (non-ocean) cells with z
above the upper bound (clip to score 0) and the share with z below the lower
bound (clip to score 1).

### 3.2 TPI score (topographic position index)

TPI = z minus mean z in a circular window of radius tpi_radius_m
(SusceptibilityConfig.tpi_radius_m = 200.0 m). The window includes the centre
cell. Edges are handled by reflect padding (scipy.ndimage uniform_filter with
mode="reflect" on a binary circular footprint).

score = clip((tpi_span_m - TPI) / (2 * tpi_span_m), 0, 1)

with SusceptibilityConfig.tpi_span_m = 10.0 m. A cell at the centre of a pit
(TPI = -tpi_span_m) scores 1.0; a cell at the centre of a peak (TPI =
+tpi_span_m) scores 0.0; a flat cell (TPI = 0) scores 0.5.

The circular footprint radius in cells is ceil(tpi_radius_m / cell_size_m).

### 3.3 Water distance score

d = Euclidean distance in metres to the nearest ocean-or-lake cell, computed
with scipy.ndimage.distance_transform_edt using sampling = cell_size_m.

score = 1 - clip(d / water_distance_max_m, 0, 1)

with SusceptibilityConfig.water_distance_max_m = 1000.0 m. A cell adjacent to
water scores near 1.0; a cell farther than 1000 m from water scores 0.0.

### 3.4 Land cover score

Lookup of runoff_index_* values by class code:

- impervious (code 0): runoff_index_impervious = 1.0
- open_ground (code 1): runoff_index_open_ground = 0.6
- vegetation (code 2): runoff_index_vegetation = 0.25
- water (code 3): runoff_index_water = 1.0

These are the same values as SusceptibilityConfig.runoff_index_*.
The land cover code mapping is defined in environment.py LAND_COVER_CODES.

### 3.5 Slope score

slope = gradient magnitude from np.gradient of the feature elevation
(rise over run, dimensionless).

score = 1 - clip(slope / slope_max, 0, 1)

with SusceptibilityConfig.slope_max = 0.3. A flat cell scores 1.0; a cell
with slope >= 0.3 scores 0.0.

---

## 4. Feature elevation

For ALL feature computations, ocean cells are set to 0.0 (MSL) to avoid fake
bathymetry slopes at the coast. Lake cells keep their bed elevation. This is a
documented modelling choice: the ocean is the boundary reservoir, so its
bathymetry should not drive susceptibility gradients on land.

---

## 5. Index

The susceptibility index is the weighted sum of the five feature scores:

index = 0.35 * elevation_score
      + 0.25 * tpi_score
      + 0.15 * water_distance_score
      + 0.15 * land_cover_score
      + 0.10 * slope_score

Weights are exactly from SusceptibilityConfig (weight_elevation=0.35,
weight_tpi=0.25, weight_water_distance=0.15, weight_land_cover=0.15,
weight_slope=0.10) and sum to 1.0. Weights are never fitted to simulator
output.

The index is defined on all non-ocean cells; ocean cells are NaN. All feature
scores and the index are in [0, 1] on non-ocean cells.

---

## 6. Experiment A (specified here, implemented in a later stage)

Target = H_peak - h_initial on LAND cells (non-ocean, non-lake).

Metrics:
- Spearman rank correlation between susceptibility index and target.
- ROC-AUC for flooded >= 0.10 m, >= 0.30 m, >= 0.50 m (Mann-Whitney rank
  formulation).
- Top-decile lift.

Baselines:
- elevation-only score.
- random permutation of the index (fixed seed).
- leave-one-feature-out ablations.

Coverage: the 5 experiment seeds from SeedConfig.env_experiment_seeds()
(10406733617104774339, 2112024614403005981, 4656971133199668420,
11861282352235076592, 6574970189797259754) x scenarios R, S1, S2, S3
(from ExperimentConfig.externals, n_env_seeds=5).

Synthetic recovery testing only: no external validity is claimed.

---

## 7. Known limitations

- The elevation transform range (0.0 to 40.0 m) clips terrain above 40 m to
  score 0; terrain reaches 60 m, so high inland nodes always score 0 on the
  elevation feature. Clip fractions are reported, not refit (weakness W10).
- TPI and slope are grid-scale proxies, not physically derived quantities.
- The simulator is a simplified water balance with terrain-blind drainage
  (see README.md Known limitations).
- The index and the simulator share one synthetic terrain generator, so
  agreement demonstrates internal consistency only.
- On seed 2027, 19.6 percent of scored cells exceed the 40 m elevation bound
  and score 0 on the elevation feature, and the index correlates 0.905 with
  the elevation score over land; the index is dominated by elevation, so an
  elevation-only baseline is expected to be competitive.

---

## 8. Layer separation test (TC4)

Structural: parse the susceptibility module source with ast and assert the set
of imported modules is limited to numpy, scipy.ndimage, dataclasses, typing,
urbanflood.config, and (for typing only) urbanflood.environment. Assert no
public function has a parameter name containing drainage, outfall, depth,
spill, or lake_spec.

Behavioural: build an altered environment (outfall_mask inverted,
outfall_invert_m shifted, lake_specs spill levels shifted) and assert
build_susceptibility(altered).index is byte-identical to the original.

---

## 9. Results (Experiment A, recorded 2026-11-13)

Experiment A compared the static susceptibility index against simulated peak
depths (target = H_peak - h_initial on LAND cells) across 5 environment seeds
x 4 scenarios (R, S1, S2, S3) = 20 seed-scenario pairs. Results are mean over
the 5 seeds; "index" = full susceptibility index with declared weights
(0.35 elevation, 0.25 TPI, 0.15 water distance, 0.15 land cover, 0.10 slope).

### auc_030 (flooded >= 0.30 m)

| Scenario | index | elevation_only | random | drop_elevation | drop_tpi | drop_water_distance | drop_land_cover | drop_slope |
|----------|-------|----------------|--------|----------------|----------|---------------------|-----------------|------------|
| R        | 0.722172 | 0.668987 | 0.500241 | 0.763644 | 0.649378 | 0.742148 | 0.734919 | 0.714618 |
| S1       | 0.741037 | 0.700976 | 0.499746 | 0.760007 | 0.673357 | 0.762694 | 0.754288 | 0.734128 |
| S2       | 0.819635 | 0.782328 | 0.497304 | 0.824937 | 0.769822 | 0.817405 | 0.835894 | 0.812675 |
| S3       | 0.724680 | 0.672852 | 0.501793 | 0.764013 | 0.652705 | 0.744220 | 0.737685 | 0.717150 |

### spearman

| Scenario | index | elevation_only | random |
|----------|-------|----------------|--------|
| R        | 0.363357 | 0.270281 | -0.004375 |
| S1       | 0.373064 | 0.278820 | -0.003778 |
| S2       | 0.423517 | 0.332318 | -0.003541 |
| S3       | 0.363685 | 0.270684 | -0.004398 |

### index vs elevation_only on auc_030 (per-seed paired)

- wins 5 of 5 in each of R, S1, S2, S3
- mean differences: R 0.053185, S1 0.040060, S2 0.037307, S3 0.051828

### lift_010 (top decile at 0.10 m threshold)

| Scenario | index | elevation_only |
|----------|-------|----------------|
| R        | 3.656844 | 4.364120 |
| S1       | 3.368170 | 4.040628 |
| S2       | 4.361623 | 5.386710 |
| S3       | 3.660798 | 4.417125 |

### lift_030 (top decile at 0.30 m threshold)

| Scenario | index | elevation_only |
|----------|-------|----------------|
| R        | 3.341657 | 3.288309 |
| S1       | 3.310246 | 3.724910 |
| S2       | 4.753902 | 5.471735 |
| S3       | 3.352162 | 3.362287 |

### Recorded statements

(a) The index ranks simulated peak depth better than the elevation-only
baseline on auc_030 in all 20 seed-scenario pairs.

(b) The elevation-only baseline has higher top-decile lift at the 0.10 m
threshold in every scenario and at 0.30 m in S1, S2 and, marginally, S3
(3.362287 vs 3.352162); the index is higher at 0.30 m only in R
(3.341657 vs 3.288309).

(c) The declared weights are not optimal: removing the elevation feature
gave a higher auc_030 than the full index in all four scenarios, and
removing TPI gave the lowest; the weights were declared before the
experiment and were NOT re-fitted, because tuning on the evaluation target
would break the leakage-free claim.

(d) The random-permutation control is near 0.5 AUC and 0 Spearman.

(e) Five environments from one generator support "consistent across seeds",
not statistical significance.

(f) Absolute AUC 0.72 to 0.82 is moderate ranking ability, not forecasting
skill; synthetic recovery only.

Elevation clip on seed 2027: 19.6 percent of scored cells above the 40 m
bound score 0 on the elevation feature.

**End of Phase 3 susceptibility specification.**

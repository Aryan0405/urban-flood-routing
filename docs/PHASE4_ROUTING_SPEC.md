# Phase 4 Routing Specification

**SYNTHETIC DATA PROJECT.** All parameters, fields, and results are synthetic.
Nothing here represents real-world Mumbai geography, roads, or flood events.
This is a routing comparison on synthetic data; it makes no claim about
real-world navigation or emergency response.

Specification: Architecture V7 (approved). Phase 4 additions recorded below are
new engineering decisions; where V7 already locks a value (grid, cell size,
speeds, thresholds, lambda sweep, OD pair count), this document preserves it
verbatim. Decisions marked "adopted (author-delegated, Stage 14)" are new
engineering tolerances and equations; they are not values recovered from the
original V7 specification.

---

## 1. Purpose and layer separation

Phase 4 implements risk-aware routing: Dijkstra on travel time with
penalized impedance and depth-threshold closures, compared against
conventional shortest paths (free-flow travel time only).

Layer 3 (routing) consumes only the road graph and depth arrays produced by
earlier layers. The routing module must NOT import or read susceptibility,
drainage, outfalls, lake_specs / spill levels, initial_state, the simulator,
or the orchestrator. The experiment runner (not the routing module) calls
the orchestrator to obtain the depth arrays and passes them to the routing
module as plain NumPy arrays.

Allowed inputs to routing:
- RoadNetwork from SyntheticEnvironment (node_rows, node_cols,
  node_elevation_m, edge_u, edge_v, edge_length_m, edge_free_flow_time_s,
  edge_class, edge_is_bridge, edge_deck_m, edge_path_cells)
- Peak edge depths from orchestrator.peak_edge_depths() (shape (n_edges, 3),
  metres, start/mid/end rule)
- Peak water-surface elevation (elevation_m + H_peak) sampled with the same
  start/mid/end rule, for bridge deck clearance computations
- RoadConfig (speeds, closure thresholds, label thresholds, lambda sweep,
  edge_depth_samples=3, n_od_pairs=20)

Forbidden inputs to routing:
- susceptibility scores
- drainage capacity, efficiency, or blocked fields
- outfall_mask, outfall_invert_m
- lake_specs / spill levels
- simulated depths h_final or h_initial
- initial_state module
- simulator or orchestrator modules (except to obtain arrays, which are then
  passed as plain NumPy arrays)

---

## 2. Effective edge depth d_e

RoadConfig does not define how d_e is computed from the three edge samples.
This section adopts (author-delegated, Stage 14) the following definition:

### Non-bridge edges
d_e = max over the three samples of peak_edge_depths (start, midpoint, end),
in metres. This is the worst-case (peak) depth along the edge.

### Bridge edges
The samples on lake cells are lake depth, not deck depth (a lake already
holds metres of water at t=0). Use:

d_e = max( peak_start_sample, peak_end_sample,
           max(0, surface_mid - deck_m) )

where:
- peak_start_sample = peak edge depth at column 0 (start cell)
- peak_end_sample = peak edge depth at column 2 (end cell)
- surface_mid = midpoint sample of the peak water-surface elevation
  (elevation_m + H_peak), sampled with the same start/mid/end rule. This is
  obtainable by calling peak_edge_depths() on an object whose H_peak is
  elevation_m + H_peak.
- deck_m = bridge deck elevation from RoadNetwork.edge_deck_m (absolute
  elevation in metres above MSL; NaN for non-bridge edges)

This is an approximation: on short bridges the midpoint can interpolate
between a land cell and a lake cell, so the sampled midpoint may not equal
any actual grid-cell depth.

If the real edge_deck_m semantics differ from absolute elevation above MSL
or NaN-for-non-bridge, adapt the formula and flag the discrepancy.

---

## 3. Closure

An edge is closed (removed from a graph copy) when d_e >= closure_threshold.

Closure thresholds:
- closure_thresholds_sweep_m = (0.20, 0.30, 0.45) (from RoadConfig)
- closure_threshold_default_m = 0.30 (from RoadConfig)

No soft penalty applies to a closed edge; closed edges are removed from the
routing graph copy before path search.

---

## 4. Penalised impedance for open edges

For open edges (d_e < closure_threshold), the cost is:

cost_e = t_e * (1 + lambda * d_e / closure_threshold)

where:
- t_e = free-flow travel time from edge length and class speed
  (edge_free_flow_time_s from RoadNetwork, computed as length * 3.6 / speed)
- d_e = effective edge depth as defined in section 2
- closure_threshold = the threshold being used for the policy

Since open edges have d_e < closure_threshold, the ratio d_e /
closure_threshold is in [0, 1).

Lambda sweep (from RoadConfig.lambda_sweep): (0, 0.5, 1, 2, 5, 10).
lambda = 0 is pure travel time (no flood penalty).

---

## 5. Shortest paths

Dijkstra (networkx) on cost_e. Build the graph with nodes and edges inserted
in canonical id order (edge ids follow canonical ordering by unordered cell
pair, as in RoadNetwork) so that ties are deterministic.

If parallel edges exist between the same node pair, keep the lowest-cost edge
and record its id. (Note: in the current RoadNetwork implementation edges are
already canonicalized by unordered cell pair, so parallel edges do not arise;
this rule is defensive.)

---

## 6. OD pairs

20 per environment (RoadConfig.n_od_pairs = 20), drawn from road nodes using
the dedicated "od_pairs" stream. The experiment runner derives the integer
stream seed via SeedConfig.stream_seed("od_pairs") (which calls
derive_stream_seed(master_seed, stream_name)) and passes that integer seed to
the routing module. The routing module itself never imports SeedConfig and
receives only the seed integer.

OD pair selection is independent of any flood result (no selection on depth).

Constraints:
- origin != destination
- straight-line distance at least 1000 m (documented modelling choice)
- no duplicate pairs
- fixed order
- the same pairs for every scenario, threshold, and lambda of that
  environment

---

## 7. Policies

- P0 = conventional shortest free-flow path on the full graph, flooding
  ignored (lambda = 0, no closures).
- P(threshold, lambda) = closure at the threshold plus penalised impedance.

For P0, all edges are open and cost = t_e (lambda = 0).

---

## 8. Per-pair metrics

All metrics are evaluated with the same d_e for both policies (so P0 is
scored on the flood it ignored). For each OD pair:

- reachable: boolean, whether a path exists under the policy
- free-flow travel time: sum of t_e along the P0 route (seconds)
- detour ratio: policy route time / P0 route time (only if both reachable)
- route length: sum of edge_length_m along the policy route (metres)
- max d_e on the route: maximum effective edge depth along the route
  (metres)
- length-weighted mean d_e (exposure): weighted by edge length (metres)
- share of route length with d_e >= label threshold: fraction of route
  length where d_e >= label_threshold (default 0.10, sweep
  (0.05, 0.10, 0.20, 0.30))
- number of edges on the route with d_e >= the policy's closure threshold:
  for P0 this is the number of P0 route edges that the closure policy would
  close; plus a flag "infeasible under this threshold" if > 0 (meaning the
  P0 route itself would be blocked by the closure policy)

---

## 9. Experiment C aggregation

Per (environment seed, scenario, threshold, lambda):
- reachable fraction: fraction of OD pairs reachable under the policy
- means over reachable pairs: detour ratio, route length, max d_e,
  length-weighted mean d_e, share of route length above label threshold,
  number of closing edges on route
- paired comparison versus P0 on pairs where both policies produce a route

Then aggregate across the five environment seeds.

Reporting language: "lower simulated flood exposure" and "lower modelled
risk"; never "safe".

---

## 10. Data source

Experiment C reuses the H_peak arrays cached by Experiment A
(hpeak_seed<seed>_<scenario>.npy; run_meta.txt records steps 720, dt 30.0)
so no simulation is re-run. The runner must refuse if run_meta.txt steps or
dt differ from 720 / 30.0, and must refuse an --out inside the repository.

---

## 11. Known limitations

- Peak depth is time-invariant and conservative: an edge counts as flooded
  for the entire trip even if it flooded briefly.
- No depth-speed relation for vehicles; the penalised impedance is a
  modelling choice, not calibrated.
- Peak depth is sampled at 3 positions (start, midpoint, end), so a flooded
  stretch between samples can be missed.
- Bridge-deck approximation: the midpoint sample on a bridge can interpolate
  between a land cell and a lake cell on short bridges.
- Free-flow speeds (50/30/20 km/h) are locked modelling choices, not
  calibrated.
- Unreachable pairs arise when origin or destination edges close under the
  closure threshold.
- Synthetic OD distribution; no real-world origin-destination data.
- Internal consistency only; no emergency-response claim.

---

## 12. Planned tests (not implemented here)

- Toy-graph shortest path: Dijkstra returns expected path and cost on a
  hand-built graph.
- Closure removes exactly the edges with d_e >= closure_threshold and keeps
  all edges with d_e < threshold.
- lambda = 0 on an unreachable-free graph equals the free-flow shortest path.
- Penalised route avoids a shallow-flooded fast edge when lambda is large
  enough, in a hand-built example.
- Bridge d_e rule on a toy bridge edge.
- Determinism: same inputs produce identical routes and costs.
- OD pair determinism and independence from depth arrays.
- Structural layer test (AST imports): routing module imports limited to
  numpy, networkx, dataclasses, typing, collections.abc, __future__, and
  urbanflood.config / urbanflood.roads for types only. No public function
  has a parameter name containing susceptibility, drainage, outfall, spill,
  lake_spec, initial_state, simulator, orchestrator, h_final, h_initial,
  h_peak, or env. State the reason: the routing module consumes plain
  per-edge arrays (including per-edge depth samples) and takes no cell-level
  arrays and no environment object, so edge depths are allowed by name.
- The runner refuses a wrong-meta cache (run_meta.txt steps/dt mismatch).

---

## 13. Module interface

The routing module (src/urbanflood/routing.py) exposes these public functions:

- effective_edge_depth(peak_samples, surface_samples, is_bridge, deck_elevation_m) -> np.ndarray (n_edges,)
  Computes effective edge depth d_e: max over the three peak samples for
  non-bridge edges; max(peak start, peak end, max(0, surface mid - deck)) for
  bridge edges.
- sample_od_pairs(node_rows, node_cols, cell_size_m, n_pairs, min_distance_m,
  seed) -> np.ndarray (n_pairs, 2) int
  Deterministically samples OD pairs of node indices using numpy RNG with the
  given seed.
- policy_costs(edge_time_s, d_e, closure_threshold_m, lam) ->
  (cost (n_edges,), open_mask (n_edges,) bool)
  Computes per-edge travel cost; returns inf cost for closed edges.
- solve_routes(n_nodes, edge_u, edge_v, cost, open_mask, od_pairs) ->
  list of RouteResult (frozen dataclass: reachable bool, edge_ids tuple[int,
  ...], node_path tuple[int, ...], total_cost float)
  Runs Dijkstra per OD pair on the open-edge graph.
- route_metrics(edge_ids, edge_length_m, edge_time_s, d_e, label_threshold_m,
  closure_threshold_m) -> dict
  Computes per-route summary metrics (length, time, max d_e, exposure, etc.).

## 14. Results (Experiment C)

Experiment C ran 5 environment seeds x 4 scenarios (R, S1, S2, S3) x closure
thresholds 0.20/0.30/0.45 x lambda 0/0.5/1/2/5/10, with 20 OD pairs per seed
(100 pooled per scenario), using cached H_peak arrays from Experiment A
(steps=720, dt=30.0). Full results are in docs/experiment_c/README.txt.

Headline (closure 0.30, lambda 0), pooled over 100 pairs:
- reachable fraction: R 0.50 (per-seed 0.25 to 0.65), S1 0.26 (0.05 to 0.55),
  S2 0.48 (0.25 to 0.60), S3 0.50 (0.25 to 0.65).
- share of P0 routes crossing a closed edge: R 0.95, S1 0.97, S2 0.95,
  S3 0.95.
- mean detour ratio, pairs where both routes exist: R 1.5251 (n=50), S1
  1.4788 (n=26), S2 1.5195 (n=48), S3 1.5344 (n=50).

Lambda effect, closure 0.30, scenario R: mean exposure reduction vs P0
0.0832 m (lambda 0) to 0.0869 m (lambda 10); mean detour 1.5251 to 1.5685;
fraction of pairs with lower exposure 0.90 to 0.94 (among pairs where both
routes exist). Reachable fraction does
not change with lambda (the closure set fixes connectivity).

Closed edges at 0.30 (of 2341 to 2357 edges) by seed order
  (10406733617104774339, 2112024614403005981, 4656971133199668420,
  11861282352235076592, 6574970189797259754):
  R 361/349/362/384/394; S1 518/465/476/506/529;
  S2 420/395/442/459/443; S3 367/356/370/389/402.

Unreachable-pair classification at 0.30, lambda 0, pooled over 5 seeds x 4
scenarios x 20 pairs = 400 cases: reachable 174; different non-trivial
components 123; origin isolated 57; destination isolated 35; both isolated 11.

Controls and diagnostics (scripts were outside the repo, not committed):
- zero-depth control: 20 of 20 pairs reachable under both P0 and the
  closure policy, cost ratio exactly 1.0, every seed and scenario.
- the max-d_e edge is the same edge with the same d_e in R and S2 for all
  five seeds.
- lake check covered two seeds and scenario R only.

Interpretation limits:
- reachable_fraction measures network connectivity under closure, not
  routing quality; about 54 percent of unreachable pairs are splits into
  non-trivial components rather than isolated endpoints.
- detour and exposure are means over pairs where both routes exist, so they
  are biased toward easier pairs.
- P0 ignores depth, so exposure reduction against P0 is largely by
  construction; the informative comparison is the lambda effect, which is
  small here.
- the shared maximum d_e appears to come from road nodes in terrain pits; roads are
  not terrain-aware, so this appears to be an artifact of the synthetic road
  layout. The road generator rule was not checked.
- synthetic recovery testing only; no real-world or Mumbai claim.

Reporting language: "lower simulated flood exposure" and "lower modelled
risk"; never "safe".

---

**End of Phase 4 routing specification.**

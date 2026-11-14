Experiment C outputs (synthetic recovery testing only)

Setup:
  5 environment seeds x 4 scenarios (R, S1, S2, S3) x closure
  thresholds 0.20/0.30/0.45 x lambda 0/0.5/1/2/5/10; 20 OD pairs per seed
  (100 pooled per scenario); cached H_peak from Experiment A (steps=720,
  dt=30.0); command: python run_experiment_c.py --expa C:\Users\Aryan\expA
  --out <dir outside the repo>; runtime about 1 to 1.5 s per seed.

Cached outputs live under docs/experiment_c/ (ASCII). No .npy file is stored
inside the repository. pairs.csv is not stored (regenerable with
run_experiment_c.py).

summary.csv: one row per (scenario, threshold_m, lam), aggregated over
seed x OD-pair rows (aggregate_policy_rows).
summary.txt: aggregated results per scenario, closure threshold and
lambda (reachable fraction, share of P0 routes crossing a closed edge,
mean detour ratio, mean exposure reduction, fraction of pairs with lower
exposure), plus a headline block at threshold 0.30 for lambda 0 and 1.
Values are pooled over all seed x OD-pair rows.
edge_summary.csv: one row per seed x scenario with closed-edge counts at
each threshold (current d_e), plus mean/max d_e.
run_meta_c.txt: seeds, scenarios, thresholds, lambdas, steps, dt,
od_min_distance_m.

Headline (closure 0.30, lambda 0), pooled over 100 pairs:
  reachable fraction: R 0.50 (per-seed 0.25 to 0.65), S1 0.26 (0.05 to 0.55),
  S2 0.48 (0.25 to 0.60), S3 0.50 (0.25 to 0.65).
  share of P0 routes crossing a closed edge: R 0.95, S1 0.97, S2 0.95, S3 0.95.
  mean detour ratio, pairs where both routes exist: R 1.5251 (n=50), S1 1.4788
  (n=26), S2 1.5195 (n=48), S3 1.5344 (n=50).

Lambda effect, closure 0.30, scenario R: mean exposure reduction vs P0 0.0832
  m (lambda 0) to 0.0869 m (lambda 10); mean detour 1.5251 to 1.5685;
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
    five seeds (consistent with ponding in pits the tide does not reach).
  - lake check (scenario R, seeds 10406733617104774339 and
    6574970189797259754 only): the max-d_e edges are edge 61 (cells (4,67),
    (5,67), elevation 34.60 and 30.52 m, H_peak 0.0018 and 3.6198 m, 30 cells
    from the nearest lake) and edge 1446 (cells (60,20), (60,21), elevation
    6.10 and 3.34 m, H_peak 0.467 and 3.225 m, 8 cells from the nearest lake);
    no lake cell, initial depth 0. Replacing d_e by H_peak minus initial depth
    (floored at 0) left closed edges (361 and 394) and reachable pairs (13/20
    and 5/20) unchanged.

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
    not terrain-aware, so this may reflect the synthetic road layout. The road generator rule was not checked.
  - lake check covered two seeds and scenario R only.
  - synthetic recovery testing only; no real-world or Mumbai claim.

All results are SYNTHETIC recovery testing only; no claim is made about
real-world flood prediction or emergency response.

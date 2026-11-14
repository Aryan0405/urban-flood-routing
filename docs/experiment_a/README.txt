Experiment A outputs (synthetic recovery testing only)

How to reproduce:
    python run_experiment_a.py --out <dir outside repo>
    20 runs (5 experiment seeds x 4 scenarios R/S1/S2/S3), steps=720, dt=30.0.
    About 3 minutes per run on the development machine.

Experiment seeds (from SeedConfig.env_experiment_seeds()):
    10406733617104774339
    2112024614403005981
    4656971133199668420
    11861282352235076592
    6574970189797259754

Cached H_peak arrays live outside the repository in <dir outside repo>/.
No .npy file is stored inside the repository.

results.csv: one row per seed x scenario x method (ASCII).
summary.txt: aggregated results per scenario x method, plus per-scenario
index vs elevation_only comparison on auc_030 (wins/ties/losses/skipped,
mean difference).

All results are SYNTHETIC recovery testing only; no claim is made about
real-world flood prediction or emergency response.

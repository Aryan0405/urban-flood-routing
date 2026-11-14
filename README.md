# Synthetic Urban Flood Simulation and Risk-Aware Routing Engine

> **SYNTHETIC DATA PROJECT.**
> This project uses **no real-world data of any kind**. There is no real Mumbai
> road network, no real terrain or digital elevation model, no real drainage or
> sewer records, no real tide gauge or rainfall records, and no real flood
> event archive. No OpenStreetMap, OSMnx, Copernicus, ESA WorldCover,
> Planetary Computer, Nominatim, paid geospatial APIs, or external map
> downloads are used. No API keys are required and no internet connection is
> needed at runtime. Every dataset, map, road, drainage field, and result
> shown in this repository or its application is **SYNTHETIC**, generated
> programmatically with fixed, documented random seeds.

An undergraduate portfolio project: a synthetic coastal urban environment
inspired by Mumbai, a physically motivated simplified flood simulator, a
leakage-free flood susceptibility index, and risk-aware routing compared
against conventional shortest paths.

## What this project is

An engineering simulation and susceptibility-modelling exercise.

## What this project is NOT

- Not a calibrated flood prediction model.
- Not a real-world Mumbai flood forecasting system.
- Not a real navigation or emergency-response product.
- Not a validated representation of actual Mumbai flood geography.

Route reporting always uses the terms "lower simulated flood exposure" and
"lower modelled risk". No route in this project is ever described as "safe".

## Architecture (V7, approved)

Three strictly separated layers over a shared synthetic environment:

| Layer | Responsibility | Inputs it must never see |
|---|---|---|
| 1. Susceptibility | static weighted index from terrain and land cover only | drainage capacity, drainage efficiency, blocked drains, outfalls, simulated depths |
| 2. Flood simulator | grid water balance with per-face sea exchange and linear head-difference exchange between cells | susceptibility scores |
| 3. Routing | Dijkstra on travel time, penalized impedance, depth-threshold closures | susceptibility layer |

Data flow: generation (terrain, land cover, hidden drainage, weather, roads)
drives layers 1 and 2; layer 2 produces peak-depth maps consumed by layer 3;
evaluation compares layers 1 and 2 after the fact (synthetic recovery
testing only; no external validity is claimed anywhere).

## Locked conventions (Architecture V7)

- **Vertical datum:** synthetic mean sea level, MSL = 0.0 m, is the single
  reference. Terrain spans [0.2, 60.0] m; the coastal land band is pinned to
  [0.2, 1.2] m; outfall inverts sit at cell bed elevation; bridge decks sit at
  the crossed waterbody level plus 1.0 m.
- **Rainfall partition:** `V_precip = R_k * A * dt` is total precipitation
  arriving at a cell; `V_runoff = alpha * V_precip` is the portion entering
  the simulated surface-water system; `V_loss = V_precip - V_runoff` is the
  portion removed before entering surface water. The depth update adds
  `V_runoff` only. Mass ledger: `IN = sum(V_precip) + SEA_IN`,
  `OUT = sum(V_loss) + sum(V_drain) + SEA_OUT`, residual asserted below 1e-9
  relative on every run.
- **Synchronous update:** one old-state snapshot per timestep feeds every
  candidate quantity (drainage, per-face sea exchange, inter-cell fluxes).
  Rainfall is applied in the final simultaneous update and cannot be drained
  or transported within the same timestep. No sequential cell updates.
- **Canonical ordering:** every inter-cell pair and sea face carries a stable
  identifier derived only from grid geometry; candidates are applied in one
  fixed canonical order for per-cell accumulation, rescaling, and ledger
  tallies; bitwise order invariance is asserted by TC14.
- **Tide:** `eta(t) = A_t * sin(2*pi*t/T)` with T = 44712 s (M2 period used as
  a timescale on a synthetic coast), zero phase, zero surge. One evaluation
  per timestep (interval midpoint) feeds both outfall blocking and coastal
  exchange.
- **Exchange coefficient:** c = 0.1 m/s is an empirical modelling choice. The
  bound `f_step = 2*c*dt/dx <= 0.20` is a declared numerical design bound, not
  a proof of stability. Empirical checks are TC8, TC10, and TC12; c is never
  tuned to pass the dt sensitivity test.
- **Scenarios:** R (reference, peak 90 mm/h, 120 mm total, A_t = 0.8 m,
  20 percent blocked), S1 (rainfall only: 150 mm/h, 200 mm), S2 (tide only:
  A_t = 1.8 m), S3 (blockage only: 40 percent blocked). Symmetric triangular
  pulses from 3600 s to 13200 s inside a 6 h window, applied by exact
  interval integration.

## Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold and configuration | complete |
| 1 | Environment generation (terrain, land cover, drainage, weather, roads) | pending |
| 2 | Flood simulator and dt sensitivity | pending |
| 3 | Susceptibility index and Experiment A | pending |
| 4 | Routing and Experiment C | pending |
| 5 | Streamlit application (5 sections) | pending |
| 6 | Closeout and documentation | pending |

See STATUS.md for the current phase log with actual test results.

## Repository layout

```
src/urbanflood/   package code (config and rng in Phase 0)
tests/            pytest suite, including the TC6 ASCII-only scan
STATUS.md         phase log
requirements.txt  minimum-version constraints (NOT exact pins)
pyproject.toml    package metadata; no dependency versions declared
```

## Environment setup (Windows PowerShell)

The project targets Python 3.11. Verified during the Phase 0 corrective
audit: Python 3.11.9 was found via the Windows py launcher and used to create
the project-local virtual environment. No global Python installation was
made and no package was installed outside the virtual environment.

```powershell
# create the environment with the py launcher (Python 3.11)
py -3.11 -m venv .venv

# activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# if activation is blocked by execution policy, run once and then retry:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# install the minimum-constraint dependency set (project-local only)
python -m pip install -r requirements.txt

# run the full test suite
python -m pytest -v
```

The environment is already present in the working copy; `.venv/` is ignored
by Git (see `.gitignore`).

## Dependencies

`requirements.txt` states **minimum-version constraints, not exact pins**:

```
numpy>=1.26, scipy>=1.10, networkx>=3.1, pandas>=2.0,
matplotlib>=3.8, streamlit>=1.30, pytest>=8.0
```

`pyproject.toml` declares no dependency versions, so the two files agree by
intention. Versions actually resolved in the audited Python 3.11.9 virtual
environment (evidence for compatibility, recorded 2026-11-09):

```
numpy==2.4.6, scipy==1.17.1, networkx==3.6.1, pandas==3.1.1,
matplotlib==3.11.2, streamlit==1.67.0, pytest==9.2.0
```

Exact pinning is deliberately not used: if pinning is later adopted, it must
be verified against this environment first, per project policy.

## Running the tests

From the repository root:

```
python -m pytest -v
```

No installation step is needed; `conftest.py` puts `src/` on the import path.

## Honest limitations (full record in later phase documentation)

Validation here is synthetic recovery testing, internal consistency between
the susceptibility index and the simulator across independent synthetic
environments. It demonstrates that the system is logically coherent; it
demonstrates nothing about real Mumbai. The exchange coefficient c is an
empirical choice, not a calibrated parameter; drainage acts as an unbounded
sink with no sewer surcharge feedback, so drain-backup flooding, a dominant
mechanism in real Mumbai events, is structurally underrepresented.

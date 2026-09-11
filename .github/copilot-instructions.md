# Copilot Instructions

## What this repo is

FDTD electromagnetic simulations (openEMS/CSXCAD) of microstrip ring
resonators on polyimide substrate, used to extract dielectric constant (Dk)
and loss tangent (tanD) vs. frequency from simulated S-parameters (the "ring
resonator method" for laminate characterization). Each numbered script
(`07_...py`, `08_...py`) is a self-contained simulation + post-processing
pipeline, not a library — there is no shared module structure.

## Running a simulation

```bash
conda activate openems
python 08_ring_resonator_polyimide_dk_loss.py
```

Requires `numpy`, `matplotlib`, `openEMS` (with Python bindings), and
`CSXCAD`, available in the `openems` conda environment. There is no test
suite or linter configured; the only way to validate a change is to run the
script and inspect the printed Dk/tanD table and the generated plot.

- Output goes to `sim_results/ring_resonator/` (openEMS field/port
  time-domain dumps plus `ring_resonator_dk_loss.png`). This directory is
  regenerated on every run — safe to delete/ignore stale contents.
- Full runs are slow (broadband 100 MHz–67 GHz FDTD sweep with
  `NrTS=200000`). When iterating on a change, prefer reasoning about the
  physics/geometry first rather than running full sweeps repeatedly.

## Script structure (each numbered script follows this pipeline)

1. **Frequency sweep + substrate params** — `f_start`/`f_stop`, nominal
   `er_nominal`/`tanD_nominal` (edit these to model a different material;
   they only seed the structure — the simulation's job is to recover them).
2. **Microstrip synthesis** — Hammerstad-Jensen used to pick trace width
   `w` for ~50Ω on the given substrate.
3. **Ring geometry** — mean radius sized so the fundamental resonance lands
   near a target frequency (`f1_target`), giving several harmonics across
   the sweep.
4. **CSXCAD structure build** — substrate/ground/feed line/ring geometry,
   lumped ports, then mesh (coarse global + fine near trace/gap/ring,
   `SmoothMeshLines`, then manual dedup of near-duplicate lines).
5. **openEMS `FDTD.Run`** — writes to `SIM_DIR`.
6. **Post-processing** — `CalcPort` → S11/S21 → notch/peak detection →
   per-harmonic Dk/tanD extraction (Q factor decomposition into
   conductor/dielectric loss) → console table → 3-panel plot saved as PNG.

## Key conventions to preserve when editing

- **Boundary conditions**: `zmin` must stay `PEC` (ground plane sits on that
  boundary); all other boundaries use `PML_8` — `MUR` is noted as
  numerically unstable for this broadband case. Do not swap these without
  re-verifying stability.
- **Port placement**: lumped ports need mesh cells on both sides of the
  probe, so ports are kept a `margin` inside the domain edge, with only
  substrate/ground (not the trace/ring metal) extended into that margin —
  metal running continuously across a port shorts its series element and
  corrupts S-parameters. This was the source of a previously-debugged issue
  (see comments referencing "example 06"); preserve the ordering of
  `mesh.AddLine(...)` + `dedupe_lines(...)` + re-adding exact port-edge
  lines afterward, since dedup can silently drift port boundaries.
- **`kappa` (dielectric loss in openEMS)** is a plain conductivity, so its
  effective loss tangent scales as `tanD_nominal * (f0_kappa / f)`, not
  constant with frequency. `f0_kappa` is deliberately the *geometric* mean
  of the sweep (not the Gaussian excitation's arithmetic-mean center
  frequency) to minimize worst-case error across a wideband, log-spaced
  sweep — don't change this without re-checking low-frequency loss accuracy.
- **Resonance detection looks for dips (notches), not peaks**, in |S21| —
  this topology (single-tap gap-coupled ring) couples energy out of the
  through line at resonance. Harmonic index `n` is computed from the
  physical resonance condition using nominal `eeff`, not by counting
  detected peaks in order (weak low-order harmonics can go undetected and
  would mislabel the rest if indexed naively).
- Extensive inline comments throughout explain *why* specific numeric
  margins/clearances/boundary choices were made from prior debugging —
  read the surrounding comment before changing a geometry/mesh/boundary
  constant, since many encode a previously-found failure mode.
- `matplotlib` uses the `Agg` backend explicitly (set before importing
  `pyplot`) to avoid hanging on `plt.show()` over X11-forwarded SSH with no
  one to close the window — keep this if adding new plotting scripts.

## Other files

- `ring_resonator_model.tex` / `.pdf` — write-up of the model and
  extraction method; update alongside the script if the method changes.
- `LICENSE` — GPLv3.

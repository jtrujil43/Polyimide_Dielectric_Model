# Polyimide Dielectric Model

FDTD electromagnetic simulation of a microstrip **ring resonator** on a thin
polyimide substrate, used to extract the substrate's relative permittivity
(Dk) and loss tangent (Df/tanD) as a function of frequency — the classic
"ring resonator method" for characterizing microwave/mm-wave laminates.

The model is built and simulated with [openEMS](https://openems.de/start/)
(via its Python bindings and [CSXCAD](https://github.com/thliebig/CSXCAD)) and
post-processed in Python/NumPy/Matplotlib.

## Overview

A microstrip ring is weakly gap-coupled to a single through feed line on a
0.1 mm polyimide substrate (nominal `er = 3.5`, `tanD = 0.008`). The
structure is excited over a single wideband Gaussian pulse (100 MHz – 67 GHz),
so every harmonic resonance of the ring shows up as a notch in the simulated
|S21| transmission response.

From the measured resonant frequencies, −3 dB bandwidths, and insertion
losses, the script extracts:

1. **Circumference resonance condition** — `n·λg(f_n) = 2π·r_mean` gives the
   effective permittivity `eeff_n` at each harmonic.
2. **Dk (er)** — inverted from `eeff_n` using the Hammerstad–Jensen
   microstrip relation for the known trace width/height.
3. **Loaded Q** from the −3 dB bandwidth of each |S21| notch, and **unloaded
   Q** via the standard insertion-loss correction.
4. **Conductor-loss Q (Qc)** computed analytically (Wheeler
   incremental-inductance rule + copper surface resistance) and removed from
   Qu to isolate the dielectric-only Q (Qd).
5. **Loss tangent (tanD)** from the dielectric filling factor
   `q = (eeff − 1)/(er − 1)`: `tanD_n = 1 / (Qd_n · q_n)`.

Results (Dk and tanD vs. frequency, alongside the raw S-parameters) are
printed to the console and plotted to a PNG.

## Repository contents

| File | Description |
|---|---|
| `07_ring_resonator_polyimide_dk_loss.py`, `08_ring_resonator_polyimide_dk_loss.py` | The simulation/extraction script (two saved iterations; currently identical) |
| `ring_resonator_model.tex` / `.pdf` | Write-up describing the model and method in detail |
| `sim_results/` | openEMS simulation output (field/port time-domain data) and the resulting `ring_resonator_dk_loss.png` plot |

## Requirements

- Python 3 with `numpy` and `matplotlib`
- [openEMS](https://openems.de/start/) with its Python bindings, and
  [CSXCAD](https://github.com/thliebig/CSXCAD)

A conda environment named `openems` is assumed by the script header (adjust
as needed for your own setup).

## Usage

```bash
conda activate openems
python 08_ring_resonator_polyimide_dk_loss.py
```

This runs the FDTD simulation (output written under `sim_results/`), prints
a per-harmonic table of extracted Dk/tanD, and saves
`sim_results/ring_resonator/ring_resonator_dk_loss.png` with:

- S11/S21 vs. frequency (resonances marked)
- Extracted Dk vs. frequency (vs. nominal input)
- Extracted loss tangent vs. frequency (vs. nominal input)

To model a different substrate, edit `er_nominal` and `tanD_nominal` near
the top of the script — these values are used only to build the structure;
the whole point of the simulation is to recover them from the S-parameters.

## License

Distributed under the terms of the GNU General Public License v3.0. See
[`LICENSE`](LICENSE) for details.

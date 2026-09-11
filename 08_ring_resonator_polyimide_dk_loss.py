"""
Example 7: Microstrip Ring Resonator on Polyimide — Dk / Loss Extraction
==========================================================================
A microstrip ring resonator, weakly gap-coupled to a single through
feed line, on a thin polyimide substrate. The ring is simulated over a
single wideband excitation (100 MHz - 67 GHz); every harmonic resonance
of the ring shows up as a transmission peak in |S21|. From the measured
resonant frequencies, -3 dB bandwidths and insertion losses, the script
extracts the substrate's relative permittivity (Dk) and loss tangent
(Df) as a function of frequency -- the classic "ring resonator method"
used for microwave/mm-wave laminate characterization.

Substrate: Polyimide (nominal er = 3.5, tanD = 0.008), 0.1 mm thick.
These are typical handbook values for polyimide (e.g. Kapton-type)
films; edit `er_nominal` / `tanD_nominal` below to match your material
datasheet -- they are only used to build the structure. The whole
point of the simulation is to recover them from the S-parameters.

Method
------
1. Circumference resonance condition:  n * lambda_g(f_n) = 2*pi*r_mean
   =>  eeff_n = ( n*C0 / (2*pi*r_mean*f_n) )^2
2. Invert the microstrip eeff(er) relation (Hammerstad-Jensen) for the
   known trace width/height to recover er_n (Dk) at each resonance.
3. Loaded Q from -3 dB bandwidth of each |S21| peak; unloaded Q via the
   standard insertion-loss correction:
        Qu = QL / (1 - 10^(-IL_dB/20))
4. Conductor-loss Q (Qc) computed analytically from microstrip
   attenuation (Wheeler incremental-inductance rule + copper surface
   resistance) and removed from Qu to isolate the dielectric Q (Qd).
5. Loss tangent from the dielectric filling factor q = (eeff-1)/(er-1):
        tanD_n = 1 / (Qd_n * q_n)

Run with:
    conda activate openems
    python 07_ring_resonator_polyimide_dk_loss.py
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: avoids hanging on plt.show()
                        # when DISPLAY is set (e.g. X11-forwarded SSH session)
                        # but nobody is present to close the plot window.
import matplotlib.pyplot as plt
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

# ── Frequency sweep ───────────────────────────────────────────────────────────
f_start = 0.1e9    # 100 MHz
f_stop  = 67.0e9   # 67 GHz
f0_exc  = (f_start + f_stop) / 2
fc_exc  = (f_stop - f_start) / 2

# Reference frequency for the (frequency-independent) dielectric kappa below.
# openEMS's "kappa" is a plain Ohmic conductivity, so the *effective* loss
# tangent it produces is tanD_nominal * (f0_kappa / f) -- NOT constant with
# frequency. Using the Gaussian excitation's center frequency f0_exc (the
# arithmetic mean, ~33.5 GHz) as this reference badly overestimates loss at
# the low end of this log-wide 100 MHz-67 GHz band (tanD_eff ~300x too high
# at 100 MHz), which was enough to make the line look almost totally lossy/
# mismatched there. The geometric mean minimizes the worst-case log-frequency
# error across the band and matches the low-loss reference already used (and
# validated) in example 06.
f0_kappa = float(np.sqrt(f_start * f_stop))

SIM_DIR = os.path.join(os.path.dirname(__file__), "sim_results", "ring_resonator")
os.makedirs(SIM_DIR, exist_ok=True)

# ── Substrate (Polyimide) — nominal values used only to build the model ─────
unit         = 1e-3    # working in mm
er_nominal   = 3.5     # polyimide relative permittivity (edit for your material)
tanD_nominal = 0.008   # polyimide loss tangent          (edit for your material)
h            = 0.1     # substrate thickness [mm] (thin polyimide film/flex laminate)
eps0         = 8.854e-12

# ── Microstrip trace width for ~50 ohm on this substrate (Hammerstad synthesis)
Z0 = 50.0
A_hj = Z0 / 60 * np.sqrt((er_nominal + 1) / 2) + (er_nominal - 1) / (er_nominal + 1) * (
    0.23 + 0.11 / er_nominal
)
WH   = 8 * np.exp(A_hj) / (np.exp(2 * A_hj) - 2)   # W/h, valid for W/h < 2 (close enough here)
w    = WH * h                                       # trace width [mm]
trace_t = 0.017                                     # 0.5 oz copper [mm]

# Effective permittivity of a straight microstrip line of this W/h (used to invert Dk later)
F_shape = (1 + 12 / WH) ** -0.5
eeff_nom = (er_nominal + 1) / 2 + (er_nominal - 1) / 2 * F_shape

# ── Ring geometry: mean radius chosen so the fundamental (n=1) resonance sits
#    near 8.5 GHz, giving ~7 harmonic resonances spread across the 67 GHz band
f1_target = 8.5e9
circumference = C0 / (f1_target * np.sqrt(eeff_nom)) / unit   # [mm]
r_mean = circumference / (2 * np.pi)                          # [mm]

gap = 0.12   # coupling gap between feed line and ring [mm] (weak coupling)

print(f"Trace width       : {w:.4f} mm")
print(f"Ring mean radius  : {r_mean:.4f} mm  (circumference {circumference:.3f} mm)")
print(f"Coupling gap      : {gap:.3f} mm")
print(f"Nominal eeff      : {eeff_nom:.4f}")

# ── Layout ────────────────────────────────────────────────────────────────────
r_out = r_mean + w / 2
r_in  = r_mean - w / 2

# Lateral clearance between the feed line's outer edge and the y=0 domain/PML
# boundary. A microstrip's fringing fields extend several substrate-heights
# beyond the trace edge; truncating that too close to an absorbing boundary
# parasitically loads the line (acts like a nearby wall), badly distorting
# its characteristic impedance and causing large, frequency-independent
# reflection. y_line = w (only ~0.5 trace-width / ~1 substrate-height of
# clearance) was found to be nowhere near enough -- use the same "several
# substrate-heights / trace-widths" rule already used for the z-direction
# open-boundary air gap.
y_clear = max(10 * h, 5 * w, 0.5)                 # [mm] lateral clearance to y=0 boundary

feed_len  = 4.0                                   # feed stub length each side [mm]
sub_l     = 2 * feed_len + 2 * r_out + 2.0         # substrate length (x)
sub_w     = y_clear + w + 2 * (r_out + gap + w) + 2.0  # substrate width (y): lower clearance + trace + ring + margin

x_c = sub_l / 2                                   # ring/line centered in x
y_line = y_clear + w / 2                          # feed line centered at this y (with adequate clearance below)
y_ring = y_line + w / 2 + gap + r_out              # ring center y (clearance measured to ring's OUTER edge)

# ── Mesh resolution ───────────────────────────────────────────────────────────
lam_min  = C0 / (f_stop * np.sqrt(er_nominal)) / unit   # min wavelength in substrate [mm]
res_coarse = lam_min / 20
res_fine   = min(w, gap) / 4                            # resolve trace/gap features

# ── Build structure ───────────────────────────────────────────────────────────
CSX  = ContinuousStructure()
FDTD = openEMS(NrTS=200000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0_exc, fc_exc)
# zmin is set to PEC (not MUR/PML) because the ground plane sits exactly on
# that boundary plane. All other boundaries use PML_8: MUR was found to be
# numerically unstable (unbounded exponential energy growth) for this
# broadband (100 MHz-67 GHz) case; PML_8 fully absorbs the outgoing waves and
# keeps the simulation stable.
FDTD.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "PML_8"])
FDTD.SetCSX(CSX)

# Margin between the ports and the outer x-boundary. A lumped port's current
# probe measures a closed H-field loop that needs a valid FDTD cell on BOTH
# sides of the probe; a port placed exactly on the outermost mesh line (the
# domain edge) has no cell on its outward side, so the measured current comes
# out as pure numerical noise (near-zero), corrupting S11/S21 even though the
# simulation itself remains stable. Keeping the port a few cells inside the
# domain (with substrate/ground extended into that margin -- NOT the feed
# line/trace itself, which must stop exactly at the port planes, else the
# continuous metal shorts across the port's series element) fixes this --
# confirmed via example 06's port-current diagnostics.
margin = 1.0  # [mm]

# Polyimide substrate -- extended slightly past the ports for port-probe validity
kappa = tanD_nominal * 2 * np.pi * f0_kappa * er_nominal * eps0
sub = CSX.AddMaterial("Polyimide", epsilon=er_nominal, kappa=kappa)
sub.AddBox(priority=0, start=[-margin, 0, 0], stop=[sub_l + margin, sub_w, h])

# Ground plane
gnd = CSX.AddMetal("GND")
gnd.AddBox(priority=10, start=[-margin, 0, 0], stop=[sub_l + margin, sub_w, 0])

# Feed line (straight through line, tangent to ring). The trace must stop
# EXACTLY at the port planes (x=0 / x=sub_l), NOT extend into the margin --
# a continuous metal sheet running straight over a lumped port's vertical
# probe shorts out the port's series R element, corrupting S11/S21 to
# near-total reflection at every frequency (only substrate/ground extend
# into the margin, confirmed working via example 06).
line = CSX.AddMetal("feed_line")
line.AddBox(
    priority=10,
    start=[0, y_line - w / 2, h],
    stop=[sub_l, y_line + w / 2, h + trace_t],
)

# Ring (annulus) built as a cylindrical shell extruded through the trace thickness
ring = CSX.AddMetal("ring")
ring.AddCylindricalShell(
    start=[x_c, y_ring, h],
    stop=[x_c, y_ring, h + trace_t],
    radius=r_mean,
    shell_width=w,
    priority=10,
)

# ── Lumped ports (both ends of the feed line) ────────────────────────────────
port1 = FDTD.AddLumpedPort(
    port_nr=1, R=50,
    start=[0, y_line - w / 2, 0],
    stop=[0, y_line + w / 2, h],
    p_dir="z", excite=1.0,
)
port2 = FDTD.AddLumpedPort(
    port_nr=2, R=50,
    start=[sub_l, y_line - w / 2, 0],
    stop=[sub_l, y_line + w / 2, h],
    p_dir="z",
)

# ── Mesh ──────────────────────────────────────────────────────────────────────
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

mesh.AddLine("x", [-margin, 0, sub_l, sub_l + margin])
mesh.AddLine("x", np.linspace(x_c - r_out - gap, x_c + r_out + gap, int(2 * (r_out + gap) / res_fine) + 1))
mesh.AddLine("x", np.linspace(0, sub_l, int(sub_l / res_coarse) + 1))
mesh.AddLine("x", np.linspace(-margin, 0, max(int(margin / res_coarse), 3) + 1))
mesh.AddLine("x", np.linspace(sub_l, sub_l + margin, max(int(margin / res_coarse), 3) + 1))

mesh.AddLine("y", [0, sub_w])
# Explicit lines at the port's exact y-boundaries -- a linspace alone is not
# guaranteed to land exactly on y_line +/- w/2 (its step size need not evenly
# divide w), which would leave the lumped port's transverse extent snapped to
# the wrong nearest mesh line and desync its integration path from the
# intended trace width/R normalization.
mesh.AddLine("y", [y_line - w / 2, y_line + w / 2])
mesh.AddLine("y", np.linspace(y_line - w / 2, y_ring + r_out, int((y_ring + r_out - y_line) / res_fine) + 1))

mesh.AddLine("z", [0, h, h + trace_t])
mesh.AddLine("z", np.linspace(0, h, 5))

# SmoothMeshLines' max_res is an absolute cap on cell size everywhere it
# inserts lines -- it does NOT taper off to a coarser size far from the
# fine region, it just keeps filling at max_res. So we only apply it to
# x/y here (open-boundary air space in z is graded manually below).
mesh.SmoothMeshLines("x", res_coarse, 1.3)
mesh.SmoothMeshLines("y", res_coarse, 1.3)

# Open-boundary air space above the trace: without this the PML/zmax
# boundary sits right on (or a fraction of a mm above) the metal, which
# either falls back to PEC (if too few z-lines exist for an 8-cell PML) or
# absorbs the near-field before it can radiate/couple properly -- both give
# grossly wrong S-parameters (near-total, frequency-independent reflection).
# Manually grade this gap with a geometrically growing sequence (ratio 1.3)
# starting from the fine trace-resolution and relaxing to a coarser cap
# once away from the trace -- letting SmoothMeshLines do this instead would
# force the ENTIRE gap to stay at the fine trace-resolution (its max_res
# cap), ballooning the z-cell count.
z_air = max(10 * h, 5 * w, 1.5)   # [mm] air clearance above the trace
z_top = h + trace_t + z_air
z_res_max = 4 * (h / 4)  # relax resolution once away from the trace
z_lines = [h + trace_t]
step = h / 4
while z_lines[-1] < z_top:
    step = min(step * 1.3, z_res_max)
    z_lines.append(z_lines[-1] + step)
z_lines[-1] = z_top  # snap last point exactly to z_top
mesh.AddLine("z", z_lines)

# Remove any near-duplicate lines (can appear where manually inserted fine
# lines nearly coincide with smoothed/auto-inserted lines): sliver cells of
# a few microns force an unstable CFL timestep, so merge anything closer
# than min_gap.
def dedupe_lines(mesh, direction, min_gap):
    lines = np.sort(np.array(mesh.GetLines(direction)))
    keep = [lines[0]]
    for x in lines[1:]:
        if x - keep[-1] >= min_gap:
            keep.append(x)
        else:
            keep[-1] = (keep[-1] + x) / 2   # merge into midpoint
    mesh.SetLines(direction, keep)

min_gap = res_fine / 2
dedupe_lines(mesh, "x", min_gap)
dedupe_lines(mesh, "y", min_gap)
dedupe_lines(mesh, "z", min_gap)

# Re-assert the port planes/edges exactly after dedup -- dedupe_lines can
# merge a manually-inserted exact line into a midpoint with its nearest fine
# neighbor if the two fall within min_gap of each other, silently drifting
# the port boundary away from the intended trace edge again (this happened
# here: the port's y-edge line landed within min_gap of the fine linspace
# grid and got averaged away). Re-adding them last guarantees the ports'
# defining coordinates are present as exact mesh lines, at the cost of a
# possible single extra thin sliver cell right at that edge (harmless).
mesh.AddLine("x", [0, sub_l])
mesh.AddLine("y", [y_line - w / 2, y_line + w / 2])

# ── Run simulation ────────────────────────────────────────────────────────────
FDTD.Run(SIM_DIR, cleanup=True, verbose=0)

# ── Post-process: S-parameters ───────────────────────────────────────────────
freq = np.arange(f_start, f_stop + 1, 10e6)   # 10 MHz steps for resonance resolution
port1.CalcPort(SIM_DIR, freq)
port2.CalcPort(SIM_DIR, freq)

s11 = port1.uf_ref / port1.uf_inc
s21 = port2.uf_ref / port1.uf_inc

s11_db = 20 * np.log10(np.abs(s11) + 1e-20)
s21_db = 20 * np.log10(np.abs(s21) + 1e-20)
s21_lin = np.abs(s21)

# ── Resonance detection (notches/dips in |S21|) ──────────────────────────────
# For this single-tap, gap-coupled ring topology, each ring resonance couples
# energy OUT of the through-line and into the ring, producing a DIP (notch) in
# |S21| at resonance rather than a peak. Confirmed empirically against the
# simulated S-parameters.
def find_peaks(y, min_prominence_db=3.0):
    """Simple local-minima (notch) finder with a minimum prominence in dB."""
    y_db = 20 * np.log10(y + 1e-20)
    idx = []
    for i in range(2, len(y) - 2):
        if y[i] < y[i - 1] and y[i] <= y[i + 1] and y[i] < y[i - 2] and y[i] < y[i + 2]:
            # local peak search on both sides to check prominence
            left_max  = np.max(y_db[max(0, i - 200):i])
            right_max = np.max(y_db[i:min(len(y_db), i + 200)])
            prominence = min(left_max, right_max) - y_db[i]
            if prominence >= min_prominence_db:
                idx.append(i)
    return idx

peak_idx = find_peaks(s21_lin)

results = []
for n_est, i_pk in enumerate(peak_idx, start=1):
    f_pk = freq[i_pk]
    pk_db = s21_db[i_pk]

    # -3 dB bandwidth around this notch (search outward until rising back
    # above pk_db+3, i.e. within 3 dB of the dip bottom)
    lo = i_pk
    while lo > 0 and s21_db[lo] < pk_db + 3:
        lo -= 1
    hi = i_pk
    while hi < len(freq) - 1 and s21_db[hi] < pk_db + 3:
        hi += 1
    if hi <= lo:
        continue
    bw3db = freq[hi] - freq[lo]
    if bw3db <= 0:
        continue

    QL = f_pk / bw3db
    IL_db = -pk_db   # insertion loss magnitude (positive dB)
    if IL_db <= 0.05:
        continue   # too close to 0 dB / unreliable
    Qu = QL / (1 - 10 ** (-IL_db / 20))

    # Harmonic index from the physical resonance condition (n*lambda_g = 2*pi*r_mean),
    # using the nominal eeff as an initial guess. NOT simply the ascending detection
    # order: at partial convergence the lowest-order (weakest-coupled) harmonics can
    # be too shallow to cross the prominence threshold and go undetected, which would
    # silently mislabel the remaining higher harmonics if n were just counted 1,2,3...
    n = max(1, round(f_pk * 2 * np.pi * (r_mean * unit) * np.sqrt(eeff_nom) / C0))
    eeff_n = (n * C0 / (2 * np.pi * (r_mean * unit) * f_pk)) ** 2

    # Invert microstrip eeff(er) for the fixed W/h of this trace to get Dk
    er_n = (2 * eeff_n - (1 - F_shape)) / (1 + F_shape)

    # Dielectric filling factor
    q_n = (eeff_n - 1) / (er_n - 1) if er_n > 1 else np.nan

    # Analytic conductor Q (Wheeler incremental-inductance rule, copper)
    sigma_cu = 5.8e7
    Rs = np.sqrt(np.pi * f_pk * 4 * np.pi * 1e-7 / sigma_cu)
    Z0_line = 50.0
    alpha_c = Rs / (Z0_line * w * unit)      # simplified microstrip conductor atten. [Np/m]
    beta = 2 * np.pi * f_pk * np.sqrt(eeff_n) / C0
    Qc = beta / (2 * alpha_c) if alpha_c > 0 else np.inf

    if Qu > 0 and Qc > Qu:
        Qd = 1 / (1 / Qu - 1 / Qc) if (1 / Qu - 1 / Qc) > 0 else np.nan
    else:
        # Analytic conductor-only Qc came out <= measured total Qu, which is
        # unphysical for a clean loss separation (conductor+dielectric+radiation
        # losses can only reduce Q below the conductor-only value). This
        # typically indicates the simulated Qu is still optimistic due to
        # incomplete time-domain convergence (spectral leakage narrows the
        # apparent linewidth). Fall back to reporting the combined (total)
        # loss directly via Qu as an upper bound on the dielectric-only Q.
        Qd = Qu

    tanD_n = 1 / (Qd * q_n) if (Qd and q_n and not np.isnan(Qd) and not np.isnan(q_n) and Qd > 0) else np.nan

    results.append(dict(n=n, f0=f_pk, QL=QL, Qu=Qu, Qc=Qc, Qd=Qd,
                         eeff=eeff_n, er=er_n, tanD=tanD_n, IL_db=IL_db))

# ── Report ────────────────────────────────────────────────────────────────────
print("\n%-3s %-10s %-8s %-8s %-8s %-8s %-8s %-8s" %
      ("n", "f0(GHz)", "IL(dB)", "QL", "Qu", "eeff", "Dk", "tanD"))
for r in results:
    print("%-3d %-10.3f %-8.2f %-8.1f %-8.1f %-8.3f %-8.3f %-8.5f" %
          (r["n"], r["f0"] / 1e9, r["IL_db"], r["QL"], r["Qu"], r["eeff"], r["er"], r["tanD"]))

if results:
    dk_vals   = [r["er"] for r in results if not np.isnan(r["er"])]
    tand_vals = [r["tanD"] for r in results if not np.isnan(r["tanD"])]
    if dk_vals:
        print(f"\nMean extracted Dk   : {np.mean(dk_vals):.3f}  (nominal input: {er_nominal})")
    if tand_vals:
        print(f"Mean extracted tanD : {np.mean(tand_vals):.5f}  (nominal input: {tanD_nominal})")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(3, 1, figsize=(9, 11))

axs[0].plot(freq / 1e9, s11_db, label="|S11| dB")
axs[0].plot(freq / 1e9, s21_db, label="|S21| dB")
for r in results:
    axs[0].axvline(r["f0"] / 1e9, color="gray", linestyle=":", alpha=0.5)
axs[0].set_xlabel("Frequency (GHz)")
axs[0].set_ylabel("Magnitude (dB)")
axs[0].set_title("Ring Resonator S-Parameters (100 MHz - 67 GHz)")
axs[0].legend()
axs[0].grid(True)

if results:
    f_ghz = [r["f0"] / 1e9 for r in results]
    axs[1].plot(f_ghz, [r["er"] for r in results], "o-")
    axs[1].axhline(er_nominal, color="gray", linestyle="--", label=f"nominal Dk={er_nominal}")
    axs[1].set_xlabel("Frequency (GHz)")
    axs[1].set_ylabel("Extracted Dk")
    axs[1].set_title("Extracted Dielectric Constant vs Frequency")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].semilogy(f_ghz, [r["tanD"] for r in results], "o-", color="tab:red")
    axs[2].axhline(tanD_nominal, color="gray", linestyle="--", label=f"nominal tanD={tanD_nominal}")
    axs[2].set_xlabel("Frequency (GHz)")
    axs[2].set_ylabel("Extracted loss tangent")
    axs[2].set_title("Extracted Loss Tangent vs Frequency")
    axs[2].legend()
    axs[2].grid(True)

plt.tight_layout()
out_png = os.path.join(SIM_DIR, "ring_resonator_dk_loss.png")
plt.savefig(out_png, dpi=150)
print(f"\nPlot saved: {out_png}")
plt.show()

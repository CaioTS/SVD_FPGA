# %%
# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from CantileverBeam import CantileverBeam
from AdaptiveOO import FIR
from aux_functions import (
    format_W, lagged_matrix, svd_report, gen_SVD_weights_branches,
    FIRSVDFilterPy, gen_wsec_wfbk_filters, plot_fft_onesided_pair,
    plot_fft_onesided_n_modes, plot_singular_values, compare_filters,
    calc_svd_utilizations, calc_svd_utilizations_teo, calc_fir_utilization,
)

# %%
# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# Physical and simulation constants for the cantilever beam experiment.
# These parameters define the beam geometry, sensor/actuator positions,
# and the sampling frequency used throughout the simulation.

fs          = 416.0   # Sampling frequency (Hz)
npoints     = 100     # Number of finite elements in the beam
beamlength  = 0.58    # Beam length (m)
beamwidth   = 0.05    # Beam width (m)
beamthickness = 0.006 # Beam thickness (m)
dampingfactors = [0.01, 0.01, 0.01, 0.01, 0.01]

perturbpos  = 30      # FE node: perturbation force input
referencepos = 75     # FE node: reference accelerometer
controlpos  = 60      # FE node: control force output
errorpos    = 95      # FE node: error accelerometer

firmem      = 6400    # FIR filter memory length (samples)
mu          = 0.01    # LMS step size

# %%
# =============================================================================
# BEAM INSTANTIATION AND NATURAL FREQUENCY REPORT
# =============================================================================
# Creates the cantilever beam model and prints its natural frequencies.
# Also plots the beam geometry with annotated sensor/actuator positions.

cbeam = CantileverBeam(
    npoints=npoints,
    width=beamwidth,
    thickness=beamthickness,
    length=beamlength,
    Tsampling=1.0 / fs,
    damp=dampingfactors,
)
cbeam.reset()

print("Natural frequencies (Hz):\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()))

# Beam geometry plot
xcoords = np.linspace(0.0, cbeam.length, npoints)
xcoords = np.concatenate((xcoords, xcoords[::-1]))
ycoords = np.array([0.0] * npoints + [beamthickness] * npoints)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xcoords, y=ycoords, fill='toself', mode='lines'))
annotations = [
    (perturbpos,  beamthickness * 1.1,  -30, "Perturbation",        "start"),
    (referencepos, beamthickness * 1.1, -50, "Accel. Measurement",  "start"),
    (controlpos,  0,                     30, "Control Force",        "end"),
    (errorpos,    0,                     50, "Error Accel.",         "start"),
]
for pos, y, ay, text, arrowside in annotations:
    fig.add_annotation(
        x=pos * beamlength / npoints, y=y,
        ax=0, ay=ay, text=text,
        showarrow=True, arrowhead=1, arrowside=arrowside
    )
fig.update_layout(
    #title="Cantilever Beam",
    xaxis_title="x (m)", yaxis_title="y (m)",
    xaxis=dict(range=[0, beamlength * 1.1]),
    yaxis=dict(range=[-(beamthickness + 0.1), beamthickness + 0.1]),
    width=600, height=350
)
fig.show()
fig.write_image("figs/cantilever_beam.pdf") 

# %%
# =============================================================================
# OPEN-LOOP VIBRATION SIMULATION
# =============================================================================
# Simulates the beam response to a sinusoidal perturbation force.
# The force starts at vibstart seconds and runs until maxtime.
# Plots both the input force and the resulting acceleration response.

maxtime  = 60.0
vibstart = 10.0
vibfreq  = 10.0   # Hz

nsteps = int(maxtime * fs)
th = np.linspace(0.0, maxtime, nsteps)
xh = 0.3 * np.sin(2 * np.pi * th * vibfreq)
xh[0:int(fs * vibstart)] = 0.0   # zero force before vibstart

cbeam.reset()
err = np.zeros(nsteps)
for k in range(nsteps):
    err[k] = cbeam.getaccelms2(referencepos)
    cbeam.setforce(perturbpos, xh[k])
    cbeam.update()

fig = px.line()
fig.add_scatter(x=th, y=xh,  name="Força (N)",       mode="lines")
fig.add_scatter(x=th, y=err, name="Aceleração (m/s²)", mode="lines")
fig.show()

# %%
# =============================================================================
# IMPULSE RESPONSE & SVD WEIGHT GENERATION
# =============================================================================
# Computes the secondary path impulse response, then runs SVD decomposition
# to obtain Vt (C_weights) and Us (R_weights) for each branch.
# Also quantizes the weights and plots them for inspection.

wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam, firmem, controlpos, errorpos, referencepos)
limit = 2000
print(wsecimpulse.shape, wfbkimpulse.shape)

t_sec = np.arange(len(wsecimpulse[:limit]))
t_fbk = np.arange(len(wfbkimpulse[:limit]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_sec, wsecimpulse[:limit], linewidth=1, label="Caminho secundário")
ax.plot(t_fbk, wfbkimpulse[:limit], linewidth=1, label="Caminho de feedback")

ax.set_xlabel("Amostras")  # or "Tempo (ms)" if using fs
ax.set_ylabel("Amplitude")
ax.set_title("Respostas ao impulso dos caminhos secundário e de feedback")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("figs/respostas_impulso.pdf")
plt.show()

# %%
# =============================================================================
# PLOT FFT OF SECONDARY AND FEEDBACK PATH
# =============================================================================

fig = plot_fft_onesided_pair(wsecimpulse, wfbkimpulse, fs=fs,
                             labels=("Secondary path", "Feedback path"),
                        title="Frequency response of the paths")

fig.update_yaxes(range=[-100, -30]) 
fig.show()
fig.write_image("figs/frequencies_responses.pdf") 
# %%
# =============================================================================
# OBTAIN RESSONANCE FREQUENCIES AND THEIR 3 DB correspondent FOR SECONDARY
# =============================================================== ==============
results = plot_fft_onesided_n_modes(wsecimpulse, fs, title="FFT", n_modes=3)
for i, r in enumerate(results, 1):
    print(f"\nMode {i}")
    print(f"Resonance frequency : {r['frequency']:.3f} Hz")
    print(f"Peak magnitude      : {r['peak_db']:.2f} dB")
    print(f"-3 dB lower         : {r['f_3db_low']:.3f} Hz")
    print(f"-3 dB upper         : {r['f_3db_high']:.3f} Hz")
    print(f"Bandwidth           : {r['bandwidth_3db']:.3f} Hz")

frequencies  = np.array([r["frequency"] for r in results])
f_3db_lower  = np.array([r["f_3db_low"] for r in results])
f_3db_higher = np.array([r["f_3db_high"] for r in results])

print(frequencies , f_3db_lower , f_3db_higher)

# %%
# =============================================================================
# CALCULATE DAMPING COEFFICIENT FOR SECONDARY PATH
# =============================================================================
damping_coef  = np.zeros_like(frequencies)

for i , (f, fl, fh) in enumerate(zip(frequencies , f_3db_lower , f_3db_higher)):
    damping_coef[i] = (fh - fl) / (2*f)

print(damping_coef) 

# %%
# =============================================================================
# OBTAIN RESSONANCE FREQUENCIES AND THEIR 3 DB correspondent FOR FEEDBACK
# =============================================================== ==============
results = plot_fft_onesided_n_modes(wfbkimpulse, fs, title="FFT", n_modes=3)
for i, r in enumerate(results, 1):
    print(f"\nMode {i}")
    print(f"Resonance frequency : {r['frequency']:.3f} Hz")
    print(f"Peak magnitude      : {r['peak_db']:.2f} dB")
    print(f"-3 dB lower         : {r['f_3db_low']:.3f} Hz")
    print(f"-3 dB upper         : {r['f_3db_high']:.3f} Hz")
    print(f"Bandwidth           : {r['bandwidth_3db']:.3f} Hz")

frequencies  = np.array([r["frequency"] for r in results])
f_3db_lower  = np.array([r["f_3db_low"] for r in results])
f_3db_higher = np.array([r["f_3db_high"] for r in results])

print(frequencies , f_3db_lower , f_3db_higher)

# %%
# =============================================================================
# CALCULATE DAMPING COEFFICIENT FOR FEEDBACK PATH
# =============================================================================
damping_coef  = np.zeros_like(frequencies)

for i , (f, fl, fh) in enumerate(zip(frequencies , f_3db_lower , f_3db_higher)):
    damping_coef[i] = (fh - fl) / (2*f)

print(damping_coef) 
# %%
# =============================================================================
# CALCULATE ACCOMMODATION TIME AND FILTER'S LENGTH FOR SECONDARY
# =============================================================================
Threshold = [0.05, 0.02, 0.01]

acc_time  = np.zeros((len(Threshold), len(frequencies)))

for t_idx, thresh in enumerate(Threshold):
    for i, f in enumerate(frequencies):
        acc_time[t_idx, i] = -(np.log(thresh)) / (2 * np.pi * damping_coef[i] * f)

print("Accommodation Times:")
for t_idx, thresh in enumerate(Threshold):
    print(f"  Threshold {thresh:.2f}: {acc_time[t_idx, :]}")
    print(f"  Filter Length      : {int(np.ceil(np.max(acc_time[t_idx, :]) * fs))} samples")

# All filter lengths in one array for easy inspection
filter_lengths = {
    thresh: int(np.ceil(np.max(acc_time[t_idx, :]) * fs))
    for t_idx, thresh in enumerate(Threshold)
}
print(f"\nFilter lengths by threshold: {filter_lengths}")

len_filter = filter_lengths[0.05]

# %%
# =============================================================================
# GENERATE SVD WITH THEORETICAL NUMBER OF BRANCHES
# =============================================================================
X = wsecimpulse[:len_filter]

n_branches = 6
C, R , S= gen_SVD_weights_branches(X, n_branches, C_size=int(np.sqrt(len_filter)), plot_S=True)

svd_filter_py = FIRSVDFilterPy(C, R)

print(C.shape, R.shape)

# %%
# --- Setup ---
mem = len_filter
wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam, mem, controlpos, errorpos, referencepos)

L = int(np.sqrt(mem)) + 1
K = mem - L + 1
print(f"mem = {mem} | L = {L} | K = {K}")

# --- Stacked vs. lagged matrices ---
wsec_stack  = format_W(wsecimpulse, L)
wsec_lagged = lagged_matrix(wsecimpulse, L)

U_stack,  S_stack,  VT_stack  = svd_report("Stacked", wsec_stack)
U_lagged, S_lagged, VT_lagged = svd_report("Lagged",  wsec_lagged)


# ── Usage ──────────────────────────────────────────────────────────
#fig = plot_singular_values(S_lagged, S_stack, normalize=True,  log_y=False)
fig = plot_singular_values(S_lagged, S_stack, normalize=False, log_y=False)
fig.update_xaxes(range=[1, 12]) 
fig.show()
fig.write_image("figs/S_values_comp.pdf") 
# %%
# =============================================================================
# COMPARE ORIGINAL FILTER AND SVD FILTER PARAMETERS
# =============================================================================
svd_mem, svd_coefs, svd_ops = calc_svd_utilizations(C, R)
fir_mem, fir_coefs, fir_ops = calc_fir_utilization(len_filter)

# Compression rates
mem_compression_exp   = (1 - svd_mem   / fir_mem)   * 100
coef_compression_exp  = (1 - svd_coefs / fir_coefs) * 100
ops_compression_exp   = (1 - svd_ops   / fir_ops)   * 100

print("\n" + "=" * 80)
print(f"{'RESOURCE UTILIZATION':^80}")
print("=" * 80)

print(
    f"{'Resource':<25}"
    f"{'SVD FIR':>15}"
    f"{'Conventional FIR':>20}"
    f"{'Compression':>18}"
)

print("-" * 80)

print(
    f"{'Memory elements':<25}"
    f"{svd_mem:>15}"
    f"{fir_mem:>20}"
    f"{mem_compression_exp:>17.2f}%"
)

print(
    f"{'Coefficients':<25}"
    f"{svd_coefs:>15}"
    f"{fir_coefs:>20}"
    f"{coef_compression_exp:>17.2f}%"
)

print(
    f"{'Operations':<25}"
    f"{svd_ops:>15}"
    f"{fir_ops:>20}"
    f"{ops_compression_exp:>17.2f}%"
)

print("=" * 80)
#%%
fs = 416

N = np.arange(500,4000,300)
svd_coefs_teo = np.zeros_like(N)
svd_mems_teo = np.zeros_like(N)
svd_ops_teo = np.zeros_like(N)

fir_coefs_teo = np.zeros_like(N)
fir_mems_teo = np.zeros_like(N)
fir_ops_teo = np.zeros_like(N)

for n in N:  
    svd_mems_teo[N == n],svd_coefs_teo[N == n], svd_ops_teo[N == n] = calc_svd_utilizations_teo(n,6)
    fir_mems_teo[N == n],fir_coefs_teo[N == n], fir_ops_teo[N == n] = calc_fir_utilization(n)
# %%
compression_coefs = 100 * (1 - svd_coefs_teo/fir_coefs_teo)
compression_mems  = 100 * (1 - svd_mems_teo /fir_mems_teo)
compression_ops   = 100 * (1 - svd_ops_teo  /fir_ops_teo)

# Create the figure
fig = go.Figure()

# Add all three traces to the same plot
fig.add_trace(go.Scatter(
    x=N, 
    y=compression_coefs, 
    mode='lines+markers', 
    name='Coefficients Compression',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=N, 
    y=compression_mems, 
    mode='lines+markers', 
    name='Memory Compression',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=N, 
    y=compression_ops, 
    mode='lines+markers', 
    name='Operations Compression',
    line=dict(width=2)
))

# Update layout
fig.update_layout(
    title='Compression Metrics Comparison',
    xaxis_title='Filter Size',
    yaxis_title='Compression Ratio (%)',
    yaxis=dict(range=[0, 100]),
    height=600,
    width=900,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.2)',
        borderwidth=1
    )
)

# Show the figure
fig.show()

# %%
# --- Theoretical curves ---
N_branches = 6
N = np.arange(500, 4000, 300)

svd = np.array([calc_svd_utilizations_teo(n, N_branches) for n in N])  # (len(N), 3)
fir = np.array([calc_fir_utilization(n) for n in N])

svd_mems_teo, svd_coefs_teo, svd_ops_teo = svd.T
fir_mems_teo, fir_coefs_teo, fir_ops_teo = fir.T

compression_mems  = 100 * (1 - svd_mems_teo  / fir_mems_teo)
compression_coefs = 100 * (1 - svd_coefs_teo / fir_coefs_teo)
compression_ops   = 100 * (1 - svd_ops_teo   / fir_ops_teo)

N_patamars = filter_lengths   # filter size used in the experiment

utilizations = {}
for thresh, n in N_patamars.items(): 
    m_svd,  c_svd, o_svd = calc_svd_utilizations_teo(n, N_branches)
    m_fir , c_fir, o_fir = calc_fir_utilization(n)
    utilizations[thresh] = {
        "Coefficients": 100 * (1 -  c_svd/c_fir), 
        "Memory":       100 * (1 - m_svd/m_fir), 
        "Operations":   100 * (1 - o_svd/o_fir)}

# Access as:
# --- Simulated values for different trhesholds 
# --- Plot ---
colors = {
    "Coefficients": "#1D9E75",
    "Memory":       "#E8833A",
    "Operations":   "#5B8DEF",
}
theory = {
    "Coefficients": compression_coefs,
    "Memory":       compression_mems,
    "Operations":   compression_ops,
}

fig = go.Figure()

for name, y in theory.items():
    # ── Theoretical curve ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=N, y=y, mode="lines",
        name=f"{name}",
        line=dict(color=colors[name], width=2),
        legendgroup=name,
        showlegend=True,
    ))
    # ── Experimental markers — one per threshold ─────────────────
    for thresh, n in N_patamars.items():
        fig.add_trace(go.Scatter(
            x=[n],
            y=[utilizations[thresh][name]],
            mode="markers",
            showlegend=False,
            name=f"thresh={thresh}",
            marker=dict(
                color=colors[name],   # explicit per-metric color
                size=12,
                symbol="diamond",
                line=dict(color="white", width=1.5)
            )
        ))

# ── Vertical lines — stagger annotation positions to avoid overlap ──
positions = ["left", "center", "right"]
for i, (thresh, n) in enumerate(N_patamars.items()):
    fig.add_vline(
        x=n,
        line=dict(color="gray", width=1, dash="dot"),
        annotation=dict(
            text=f"τ={thresh}",   # shorter text
            textangle=-90,         # rotate to save horizontal space
            font=dict(size=10),
            align=positions[i % len(positions)]
        )
    )

fig.update_layout(
    title=f"SVD FIR vs FIR Compression — {N_branches} branches",
    xaxis_title="Filter size N",
    yaxis_title="Compression (%)",
    yaxis=dict(range=[0, 100]),
    template="plotly_white",
    hovermode="x unified"
    #legend=dict(groupclick="toggleitem")
)
fig.show()
fig.write_image("figs/Parameters_comp.pdf")

# %%
#Compare filter outputs 

svd_filter_py
fir_filter_py = FIR(wsecimpulse)


fig, err = compare_filters(svd_filter_py, fir_filter_py,
                           n_samples=len(wsecimpulse), fs=fs,
                           ref=wsecimpulse)

fig.write_image("figs/error_svd_fir.pdf") 


#%% Feedback Results
# %%
# =============================================================================
# GENERATE SVD WITH THEORETICAL NUMBER OF BRANCHES
# =============================================================================
X_f = wfbkimpulse[:len_filter]

n_branches = 6
C_f, R_f , S_f = gen_SVD_weights_branches(X_f, n_branches, C_size=int(np.sqrt(len_filter)), plot_S=True)

svd_fbk_filter_py = FIRSVDFilterPy(C, R)

print(C.shape, R.shape)

# %%
# --- Setup ---
mem = len_filter
wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam, mem, controlpos, errorpos, referencepos)

L = int(np.sqrt(mem)) + 1
K = mem - L + 1
print(f"mem = {mem} | L = {L} | K = {K}")

# --- Stacked vs. lagged matrices ---
wfbk_stack  = format_W(wfbkimpulse, L)
wfbk_lagged = lagged_matrix(wfbkimpulse, L)

U_stack_fbk,  S_stack_fbk,  VT_stack_fbk  = svd_report("Stacked", wfbk_stack)
U_lagged_fbk, S_lagged_fbk, VT_lagged_fbk = svd_report("Lagged",  wfbk_lagged)


fig = plot_singular_values(S_lagged_fbk, S_stack_fbk,normalize=False,log_y=False)
fig.update_xaxes(range=[1, 12]) 
fig.show()
fig.write_image("figs/S_values_comp_fbk.pdf")
# %%
#Compare filter outputs 
 
fir_filter_py = FIR(wsecimpulse)


fig, err = compare_filters(svd_fbk_filter_py, fir_filter_py,
                           n_samples=len(wfbkimpulse), fs=fs,
                           ref=wfbkimpulse)

fig.write_image("figs/error_svd_fir_fbk.pdf") 


# %%
# Joint SVD Tryouts

def compute_optimal_angle(Q):
    """
    Compute optimal Givens rotation angle from a 2x2 matrix Q.
    Equation (19) from the paper.
    """
    q11, q12 = Q[0, 0], Q[0, 1]
    q21, q22 = Q[1, 0], Q[1, 1]

    mu = 0.5 * np.arctan2(2 * q12, q11 - q22)

    # Choose l in {0,1} so second derivative is negative
    if (q22 - q11) * np.cos(2 * mu) - 2 * q12 * np.sin(2 * mu) > 0:
        mu += np.pi / 2

    return mu


def compute_Q_matrices(matrices, i, j, rotation_type):
    """
    Compute Q matrices for Givens rotation optimization.
    Equations (28-31) and (34) from the paper.

    rotation_type:
        'real'      → equations (28, 29)
        'imaginary' → equations (30, 31)
        'phase_i'   → equation  (34) for index i
        'phase_j'   → equation  (34) for index j
    """
    K = len(matrices)

    if rotation_type in ('real', 'imaginary'):
        C_gamma = np.zeros((K, 2))
        C_phi   = np.zeros((K, 2))

        for s, M in enumerate(matrices):
            if rotation_type == 'real':
                # Equations (28, 29)
                C_gamma[s, :] = [np.real(M[j, j] + M[i, i]),
                                 -np.real(M[i, j] - M[j, i])]
                C_phi[s, :]   = [np.real(M[j, j] - M[i, i]),
                                 -np.real(M[i, j] + M[j, i])]
            else:
                # Equations (30, 31)
                C_gamma[s, :] = [np.real(M[j, j] + M[i, i]),
                                 -np.imag(M[i, j] + M[j, i])]
                C_phi[s, :]   = [np.real(M[j, j] - M[i, i]),
                                 -np.imag(M[i, j] - M[j, i])]

        Q_gamma = C_gamma.T @ C_gamma
        Q_phi   = C_phi.T   @ C_phi
        return Q_gamma, Q_phi

    else:
        # Phase corrections — equation (34)
        idx = i if rotation_type == 'phase_i' else j
        C_p = np.zeros((K, 2))
        for s, M in enumerate(matrices):
            C_p[s, :] = [np.real(M[idx, idx]), -np.imag(M[idx, idx])]
        Q_p = C_p.T @ C_p
        return Q_p


def apply_givens_rotation(matrices, U, V, i, j, theta, alpha, lam=0):
    """
    Apply Givens rotation to all matrices Mk, U and V.
    Equations (10), (26), (27) from the paper.
    lam=0     → real rotation
    lam=pi/2  → imaginary rotation
    """
    N  = matrices[0].shape[0]
    K  = len(matrices)

    # Left rotation matrix Rl,ij(theta, lam) — eq (24)
    Rl = np.eye(N, dtype=complex)
    Rl[i, i] =  np.cos(theta)
    Rl[j, j] =  np.cos(theta)
    Rl[i, j] = -np.exp( 1j * lam) * np.sin(theta)
    Rl[j, i] =  np.exp(-1j * lam) * np.sin(theta)

    # Right rotation matrix Rr,ij(alpha, lam)
    Rr = np.eye(N, dtype=complex)
    Rr[i, i] =  np.cos(alpha)
    Rr[j, j] =  np.cos(alpha)
    Rr[i, j] = -np.exp( 1j * lam) * np.sin(alpha)
    Rr[j, i] =  np.exp(-1j * lam) * np.sin(alpha)

    # Update all matrices: Mk = Rl^H @ Mk @ Rr — eq (26,27)
    for k in range(K):
        matrices[k] = Rl.conj().T @ matrices[k] @ Rr

    # Update U and V — eq (10)
    U = U @ Rl
    V = V @ Rr

    return matrices, U, V


def apply_phase_correction(matrices, U, V, idx, theta):
    """
    Apply phase correction Ti(theta) — equations (32, 33).
    """
    N = matrices[0].shape[0]
    K = len(matrices)

    Ti = np.eye(N, dtype=complex)
    Ti[idx, idx] = np.exp(1j * theta)

    for k in range(K):
        matrices[k] = Ti.conj().T @ matrices[k] @ Ti

    U = U @ Ti
    V = V @ Ti.conj()

    return matrices, U, V


def jsvd_criterion(matrices):
    """
    Compute the CILS + CRD criterion — equation (3).
    Lower is better.
    """
    total = 0
    for M in matrices:
        N = M.shape[0]
        mask = ~np.eye(N, dtype=bool)
        # Off-diagonal sum (CILS)
        total += np.sum(np.abs(M[mask]) ** 2)
        # Imaginary diagonal sum (CRD addition)
        total += np.sum(np.imag(np.diag(M)) ** 2)
    return total


def ci_jsvd(matrices_in, max_sweeps=100, tol=1e-10, verbose=True):
    """
    Complex Implementation Joint SVD (CI-JSVD).
    Algorithm 1 from the paper.

    Args:
        matrices_in : list of K square complex numpy arrays (N x N)
        max_sweeps  : maximum number of sweeps
        tol         : convergence tolerance on criterion
        verbose     : print criterion per sweep

    Returns:
        U  : (N, N) left  unitary matrix
        V  : (N, N) right unitary matrix
        Dk : list of K diagonal matrices
    """
    # Deep copy so we don't modify originals
    matrices = [M.copy().astype(complex) for M in matrices_in]
    N = matrices[0].shape[0]
    K = len(matrices)

    U = np.eye(N, dtype=complex)
    V = np.eye(N, dtype=complex)

    prev_criterion = np.inf

    for sweep in range(max_sweeps):

        # ── Sweep over all index pairs (i, j) ──────────────────────
        for i in range(N):
            for j in range(i + 1, N):

                # ── Real rotations (lambda = 0) ───────────────────
                Q_gamma_r, Q_phi_r = compute_Q_matrices(matrices, i, j, 'real')
                phi_r   = compute_optimal_angle(Q_phi_r)
                gamma_r = compute_optimal_angle(Q_gamma_r)
                theta_r = (phi_r + gamma_r) / 2
                alpha_r = (phi_r - gamma_r) / 2
                matrices, U, V = apply_givens_rotation(
                    matrices, U, V, i, j, theta_r, alpha_r, lam=0)

                # ── Imaginary rotations (lambda = pi/2) ───────────
                Q_gamma_i, Q_phi_i = compute_Q_matrices(matrices, i, j, 'imaginary')
                phi_i   = compute_optimal_angle(Q_phi_i)
                gamma_i = compute_optimal_angle(Q_gamma_i)
                theta_i = (phi_i + gamma_i) / 2
                alpha_i = (phi_i - gamma_i) / 2
                matrices, U, V = apply_givens_rotation(
                    matrices, U, V, i, j, theta_i, alpha_i, lam=np.pi / 2)

                # ── Phase corrections ─────────────────────────────
                Q_pi = compute_Q_matrices(matrices, i, j, 'phase_i')
                Q_pj = compute_Q_matrices(matrices, i, j, 'phase_j')
                theta_pi = compute_optimal_angle(Q_pi)
                theta_pj = compute_optimal_angle(Q_pj)
                matrices, U, V = apply_phase_correction(matrices, U, V, i, theta_pi)
                matrices, U, V = apply_phase_correction(matrices, U, V, j, theta_pj)

        # ── Check convergence ─────────────────────────────────────
        criterion = jsvd_criterion(matrices)
        if verbose:
            print(f"Sweep {sweep+1:3d} | Criterion = {criterion:.6e}")

        if abs(prev_criterion - criterion) < tol:
            print(f"Converged at sweep {sweep+1}")
            break
        prev_criterion = criterion

    # ── Sign correction — ensure non-negative diagonal entries ────
    for k in range(K):
        diag_signs = np.sign(np.real(np.diag(matrices[k])))
        diag_signs[diag_signs == 0] = 1
        S = np.diag(diag_signs)
        matrices[k] = S @ matrices[k]
        U = U @ S

    Dk = [np.diag(np.real(np.diag(M))) for M in matrices]

    return U, V, Dk

def reconstruct(U, S, V):
    """Reconstruct W = U @ diag(S) @ V.T"""
    return U @ np.diag(S) @ V.T

def reconstruction_error(W_orig, W_approx):
    err = np.linalg.norm(W_orig - W_approx, 'fro') / np.linalg.norm(W_orig, 'fro')
    return 100 * err

print(wsecimpulse.shape, wfbkimpulse.shape)


# Format both filters as matrices
W_sec_mat = format_W(wsecimpulse[:1369], C=37)   # (R, C)
W_fbk_mat = format_W(wfbkimpulse[:1369], C=37)   # (R, C)

# JSVD only works on square matrices per the paper
# So we work on the covariance matrices instead
#C_sec = W_sec_mat @ W_sec_mat.T   # (R, R) square
#C_fbk = W_fbk_mat @ W_fbk_mat.T   # (R, R) square

C_sec = W_sec_mat
C_fbk = W_fbk_mat

# Run CI-JSVD on both covariance matrices jointly
U, V, Dk = ci_jsvd([C_sec, C_fbk], max_sweeps=50)

D_sec = Dk[0]
D_fbk = Dk[1]

print(f"Shared U : {U.shape}")      # (R, R)
print(f"Shared V : {V.shape}")      # (R, R)
print(f"S_sec    : {np.diag(D_sec)[:5]}")
print(f"S_fbk    : {np.diag(D_fbk)[:5]}")

#%%
W_sec_approx = reconstruct(U, D_sec, V)
W_fbk_approx = reconstruct(U, D_fbk, V)

print(f"W_sec_pprox  : {W_sec_approx.shape}") 
print(f"W_fbk_approx : {W_fbk_approx.shape}")   

print(f"Secondary error : {reconstruction_error(W_sec_mat, W_sec_approx):.2f}%")
print(f"Feedback error  : {reconstruction_error(W_fbk_mat, W_fbk_approx):.2f}%")
# %%

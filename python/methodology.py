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
    choose_RC,
)

# %%
# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# Physical and simulation constants for the cantilever beam experiment.
# These parameters define the beam geometry, sensor/actuator positions,
# and the sampling frequency used throughout the simulation.

fs          = 416.0 * 2  # Sampling frequency (Hz)
npoints     = 100     # Number of finite elements in the beam
beamlength  = 0.58    # Beam length (m)
beamwidth   = 0.05    # Beam width (m)
beamthickness = 0.006 # Beam thickness (m)
dampingfactors = [0.01, 0.01, 0.01]

perturbpos  = 30      # FE node: perturbation force input
referencepos = 75     # FE node: reference accelerometer
controlpos  = 60      # FE node: control force output
errorpos    = 95      # FE node: error accelerometer

firmem      = 6400    # FIR filter memory length (samples)
mu          = 0.01    # LMS step size

# Delay of the paths (samples). The first n_delay samples of the impulse
# responses do not follow the sum of exponentials h[n] = sum c_k z_k^n, so
# they add up to n_delay extra singular values (rank <= 2M + n_delay).
# They are NOT zero (h[1] is large, specially in the feedback path), so they
# cannot just be dropped: they are kept as a short direct FIR ("head") and the
# SVD is applied only to the tail h[n_delay:], delayed by n_delay samples:
#   y[n] = sum_{k<d} h[k] x[n-k]  +  SVD{h[d:]}(x[n-d])
# (see the "EFFECT OF THE DELAY" section at the end).
n_delay     = 2

# Choice of R (rows) and C (columns) of the stacked matrix, N_svd = R*C:
#   "pad"    : C = floor(sqrt N), R = ceil(N/C) and zeros appended up to R*C
#              (original behaviour; the zeros break the sum-of-exponentials
#              structure of the last column and add residual singular values)
#   "extend" : smallest R*C >= N with R/C <= rc_max_ratio; the extra R*C - N
#              taps are TRUE samples of h (filter slightly longer, no zeros)
#   "factor" : R*C = N exactly (divisor of N closest to sqrt N); falls back to
#              "extend" when N has no divisor with R/C <= rc_max_ratio
rc_mode      = "extend"
rc_max_ratio = 1.5

# %%
# =============================================================================
# HEAD + DELAYED TAIL FILTER
# =============================================================================
class HeadTailFilter:
    """
    y[n] = sum_{k<d} head[k] x[n-k] + tail(x[n-d]),  d = len(head).
    head = h[:d] (direct FIR), tail = SVD filter of h[d:]. The d-sample buffer
    is shared: it is the head FIR memory and the delay line of the tail.
    Resources added to the SVD filter: d coefs, 2d memory, 2d operations.
    """
    def __init__(self, inner, head):
        self.inner = inner
        self.head  = np.asarray(head, dtype=float)
        self.d     = len(self.head)
        self.buf   = np.zeros(self.d)

    def reset(self):
        self.buf[:] = 0.0
        self.inner.reset()

    def filterstep(self, xn):
        if self.d == 0:
            return self.inner.filterstep(xn)
        xd = self.buf[-1]                 # x[n-d]
        self.buf[1:] = self.buf[:-1]
        self.buf[0]  = xn                 # buf = [x[n], ..., x[n-d+1]]
        return self.head @ self.buf + self.inner.filterstep(xd)

    def filter(self, x):
        return np.array([self.filterstep(xn) for xn in x])


def head_resources(d):
    """Memory, coefficients and operations added by the d-tap head."""
    return 2 * d, d, 2 * d

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
    nmodes=3
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
# The SVD part models h[n_delay:len_total]; the first n_delay samples are the
# direct FIR head (HeadTailFilter). R and C come from choose_RC (rc_mode):
# with "extend" len_total >= len_filter and no zeros are padded.
R_rows, C_cols, len_svd, n_zeros = choose_RC(len_filter - n_delay, rc_mode, rc_max_ratio)
len_total = n_delay + len_svd
print(f"rc_mode = {rc_mode} | R x C = {R_rows} x {C_cols} = {R_rows * C_cols} | "
      f"samples of h = {len_svd} | zeros padded = {n_zeros} | "
      f"len_total = {len_total} (len_filter = {len_filter})")
X = wsecimpulse[n_delay:len_total]

n_branches = 6
C, R , S= gen_SVD_weights_branches(X, n_branches, C_size=C_cols, plot_S=False)

svd_filter_py = HeadTailFilter(FIRSVDFilterPy(C, R), wsecimpulse[:n_delay])

print(f"n_delay = {n_delay} | len_svd = {len_svd}", C.shape, R.shape)

# %%
# --- Setup ---
mem = len_total
wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam, mem, controlpos, errorpos, referencepos)

# Matrices are built from the delay-free part h[n_delay:], with the same R x C
wsec_tail = wsecimpulse[n_delay:]
L = R_rows
K = len(wsec_tail) - L + 1
print(f"mem = {mem} | n_delay = {n_delay} | L = {L} | K = {K}")

# --- Stacked vs. lagged matrices ---
wsec_stack  = format_W(wsec_tail, C_cols)
wsec_lagged = lagged_matrix(wsec_tail, L)

U_stack,  S_stack,  VT_stack  = svd_report("Stacked", wsec_stack)
U_lagged, S_lagged, VT_lagged = svd_report("Lagged",  wsec_lagged)


# ── Usage ──────────────────────────────────────────────────────────
#fig = plot_singular_values(S_lagged, S_stack, normalize=True,  log_y=False)
fig = plot_singular_values(S_lagged, S_stack, normalize=False, log_y=True)
fig.update_xaxes(range=[1, 12]) 
fig.show()
fig.write_image("figs/S_values_comp.pdf") 
# %%
# =============================================================================
# COMPARE ORIGINAL FILTER AND SVD FILTER PARAMETERS
# =============================================================================
svd_mem, svd_coefs, svd_ops = calc_svd_utilizations(C, R)
h_mem, h_coefs, h_ops = head_resources(n_delay)   # n_delay-tap direct FIR head
svd_mem, svd_coefs, svd_ops = svd_mem + h_mem, svd_coefs + h_coefs, svd_ops + h_ops
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
    svd_mems_teo[N == n],svd_coefs_teo[N == n], svd_ops_teo[N == n] = calc_svd_utilizations_teo(n - n_delay,6, rc_mode, rc_max_ratio)
    svd_mems_teo[N == n]  += head_resources(n_delay)[0]   # n_delay-tap head
    svd_coefs_teo[N == n] += head_resources(n_delay)[1]
    svd_ops_teo[N == n]   += head_resources(n_delay)[2]
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

# SVD part models n - n_delay coefficients; the n_delay-tap head is added on top
svd = np.array([calc_svd_utilizations_teo(n - n_delay, N_branches, rc_mode, rc_max_ratio) for n in N])  # (len(N), 3)
svd = svd + np.array(head_resources(n_delay))
fir = np.array([calc_fir_utilization(n) for n in N])

svd_mems_teo, svd_coefs_teo, svd_ops_teo = svd.T
fir_mems_teo, fir_coefs_teo, fir_ops_teo = fir.T

compression_mems  = 100 * (1 - svd_mems_teo  / fir_mems_teo)
compression_coefs = 100 * (1 - svd_coefs_teo / fir_coefs_teo)
compression_ops   = 100 * (1 - svd_ops_teo   / fir_ops_teo)

N_patamars = filter_lengths   # filter size used in the experiment

utilizations = {}
for thresh, n in N_patamars.items(): 
    m_svd,  c_svd, o_svd = np.add(calc_svd_utilizations_teo(n - n_delay, N_branches, rc_mode, rc_max_ratio),
                                  head_resources(n_delay))
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
X_f = wfbkimpulse[n_delay:len_total]   # delay-free part, same R x C as the secondary path

n_branches = 6
C_f, R_f , S_f = gen_SVD_weights_branches(X_f, n_branches, C_size=C_cols, plot_S=False)

svd_fbk_filter_py = HeadTailFilter(FIRSVDFilterPy(C_f, R_f), wfbkimpulse[:n_delay])

print(f"n_delay = {n_delay} | len_svd = {len_svd}", C_f.shape, R_f.shape)


# %%
# --- Setup ---
mem = len_total
wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam, mem, controlpos, errorpos, referencepos)

wfbk_tail = wfbkimpulse[n_delay:]
L = R_rows
K = len(wfbk_tail) - L + 1
print(f"mem = {mem} | n_delay = {n_delay} | L = {L} | K = {K}")

# --- Stacked vs. lagged matrices ---
wfbk_stack  = format_W(wfbk_tail, C_cols)
wfbk_lagged = lagged_matrix(wfbk_tail, L)

U_stack_fbk,  S_stack_fbk,  VT_stack_fbk  = svd_report("Stacked", wfbk_stack)
U_lagged_fbk, S_lagged_fbk, VT_lagged_fbk = svd_report("Lagged",  wfbk_lagged)


fig = plot_singular_values(S_lagged_fbk, S_stack_fbk,normalize=False,log_y=True)
fig.update_xaxes(range=[1, 12]) 
fig.show()
fig.write_image("figs/S_values_comp_fbk.pdf")
# %%
#Compare filter outputs 
 
fir_fbk_filter_py = FIR(wfbkimpulse)


fig, err = compare_filters(svd_fbk_filter_py, fir_fbk_filter_py,
                           n_samples=len(wfbkimpulse), fs=fs,
                           ref=wfbkimpulse)

fig.write_image("figs/error_svd_fir_fbk.pdf")


# %%
# =============================================================================
# EFFECT OF THE DELAY (n_delay) ON RANK, SVD FILTER ERROR AND RESOURCES
# =============================================================================
# For each head length d = 0..5, on both paths:
#   - Singular values of the Hankel matrix of the tail h[d:] -> numerical rank
#   - FIR-SVD with B = 2M = 6 branches on h[d:], compared with the full FIR h:
#       "head kept"    : HeadTailFilter (d direct taps + delayed SVD tail)
#       "head dropped" : first d samples simply discarded (only delay line)
#   - Coefficients of the head-kept FIR-SVD
# Expected: rank = 2M + (n_delay - d) for d < n_delay and 2M from d = n_delay
# on, where 6 branches become exact only if the head is kept.
from plotly.subplots import make_subplots
from aux_functions import run_filter

d_values  = list(range(0, 6))
rank_tol  = 1e-8        # sigma_k / sigma_1 below this is treated as zero
B_test    = 6           # 2M branches
# Longer responses, so that "extend" always has true samples available
from aux_functions import get_impulse_response
len_long  = len_filter + 4 * int(np.sqrt(len_filter))
path_pt   = {"Secondary": "Caminho secundário", "Feedback": "Caminho de feedback"}
paths     = {"Secondary": get_impulse_response(cbeam, len_long, controlpos, errorpos),
             "Feedback":  get_impulse_response(cbeam, len_long, controlpos, referencepos)}

def snr_db(y, h):
    return 10 * np.log10(np.sum(h ** 2) / (np.sum((y - h) ** 2) + 1e-30))

def svd_head_tail(h_long, d, n_fir, B, mode):
    """
    FIR-SVD of an n_fir-tap response with a d-tap head, R x C chosen by mode.
    Returns the reference h (length of the implemented filter), the outputs
    with head kept / dropped, the stacked-matrix singular values and R, C.
    """
    R_, C_, n_used, n_zeros = choose_RC(n_fir - d, mode, rc_max_ratio)
    h    = h_long[:d + n_used]
    tail = h[d:]
    C_d, R_d, S_st = gen_SVD_weights_branches(tail, B, C_size=C_, verbose=False)
    x    = np.zeros(len(h)); x[0] = 1.0
    y_kept = run_filter(HeadTailFilter(FIRSVDFilterPy(C_d, R_d), h[:d]), x)
    y_drop = run_filter(HeadTailFilter(FIRSVDFilterPy(C_d, R_d), np.zeros(d)), x)
    return h, y_kept, y_drop, S_st, R_, C_, n_zeros

delay_results = {name: {"rank": [], "snr_kept": [], "snr_drop": [], "coefs": [], "S": []}
                 for name in paths}

for name, h_long in paths.items():
    print(f"{name}: first samples h[0:5] / max|h| =", np.round(h_long[:5] / np.max(np.abs(h_long)), 3))
    for d in d_values:
        tail = h_long[d:len_filter]

        # Rank of the Hankel matrix of the tail
        L_d = int(np.sqrt(len(tail))) + 1
        S_d = np.linalg.svd(lagged_matrix(tail, L_d), compute_uv=False)
        delay_results[name]["S"].append(S_d / S_d[0])
        delay_results[name]["rank"].append(int(np.sum(S_d / S_d[0] > rank_tol)))

        # FIR-SVD of the tail (R x C from rc_mode), with and without the d-tap head
        h, y_kept, y_drop, _, R_, C_, _ = svd_head_tail(h_long, d, len_filter, B_test, rc_mode)
        delay_results[name]["snr_kept"].append(snr_db(y_kept, h))
        delay_results[name]["snr_drop"].append(snr_db(y_drop, h))
        delay_results[name]["coefs"].append(B_test * (R_ + C_) + d)

# --- Table ---
W = 78
print("\n" + "=" * W)
print(f"{'EFFECT OF THE HEAD LENGTH d (FIR-SVD, %d branches, rc_mode = %s)' % (B_test, rc_mode):^{W}}")
print("=" * W)
for name in paths:
    print(f"{name} path (FIR: {len_filter} coefs)")
    print(f"{'d':>4}{'rank':>8}{'SNR head kept':>18}{'SNR head dropped':>20}{'coefs':>10}")
    for i, d in enumerate(d_values):
        r = delay_results[name]
        print(f"{d:>4}{r['rank'][i]:>8}{r['snr_kept'][i]:>15.2f} dB{r['snr_drop'][i]:>17.2f} dB"
              f"{r['coefs'][i]:>10}" + ("   <- n_delay" if d == n_delay else ""))
    print("-" * W)
print("=" * W)

# --- Singular values for each d (both paths) ---
fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=[path_pt[name] for name in paths])
for col, name in enumerate(paths, 1):
    for i, d in enumerate(d_values):
        S_norm = delay_results[name]["S"][i][:12]
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(S_norm) + 1), y=S_norm,
            mode="lines+markers", name=f"d = {d}",
            legendgroup=f"d{d}", showlegend=(col == 1),
            line=dict(width=3 if d == n_delay else 1, color=px.colors.qualitative.Plotly[i]),
            marker=dict(color=px.colors.qualitative.Plotly[i]),
        ), row=1, col=col)
    fig.add_vline(x=B_test, line=dict(color="gray", dash="dot"), row=1, col=col)
fig.update_yaxes(type="log", range=[-17, 0.5])          # all subplots (shared axes do not copy the type)
fig.update_yaxes(title_text="σ<sub>k</sub> / σ<sub>1</sub>", row=1, col=1)
fig.update_xaxes(title_text="Índice k")
fig.update_layout(template="plotly_white", width=1000, height=450)
fig.show()
fig.write_image("figs/S_values_delay.pdf")

# --- SNR of the FIR-SVD with 6 branches vs d (head kept vs dropped) ---
fig = go.Figure()
for name, color in zip(paths, ("#1D9E75", "#E8833A")):
    fig.add_trace(go.Scatter(x=d_values, y=delay_results[name]["snr_kept"],
                             mode="lines+markers", name=f"{path_pt[name]} — cabeça mantida",
                             line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=d_values, y=delay_results[name]["snr_drop"],
                             mode="lines+markers", name=f"{path_pt[name]} — cabeça descartada",
                             line=dict(color=color, width=1.5, dash="dash")))
fig.add_vline(x=n_delay, line=dict(color="gray", dash="dot"),
              annotation=dict(text=f"d = {n_delay}"))
fig.update_layout(xaxis_title="Comprimento da cabeça d (amostras)",
                  yaxis_title=f"SER do FIR-SVD com {B_test} ramos (dB)",
                  template="plotly_white", width=750, height=450)
fig.show()
fig.write_image("figs/snr_delay.pdf")


# %%
# =============================================================================
# EFFECT OF THE ZERO PADDING (rc_mode) ON THE STACKED MATRIX AND SVD ERROR
# =============================================================================
# Same filter (head of n_delay taps, 6 branches), three choices of R x C:
#   pad    -> R*C > N, zeros at the end of the last column
#   factor -> R*C = N exactly (or "extend" if N has no good divisor)
#   extend -> R*C >= N filled with true samples of h
# The zeros cut the exponential decay inside the last column, so the stacked
# matrix gets extra (small) singular values beyond 2M and 6 branches are no
# longer exact. Without padding, sigma_7 should drop to round-off level.
rc_modes = ["pad", "factor", "extend"]
pad_results = {name: {} for name in paths}

for name, h_long in paths.items():
    for mode in rc_modes:
        h, y_kept, _, S_st, R_, C_, n_zeros = svd_head_tail(h_long, n_delay, len_filter,
                                                             B_test, mode)
        pad_results[name][mode] = dict(
            R=R_, C=C_, zeros=n_zeros, taps=len(h), S=S_st / S_st[0],
            sigma7=S_st[B_test] / S_st[0], snr=snr_db(y_kept, h),
            coefs=B_test * (R_ + C_) + n_delay)

W = 92
print("\n" + "=" * W)
print(f"{'EFFECT OF THE ZERO PADDING (d = %d, %d branches, FIR needs %d taps)' % (n_delay, B_test, len_filter):^{W}}")
print("=" * W)
print(f"{'path':<11}{'mode':<8}{'R x C':>10}{'zeros':>7}{'taps':>7}{'coefs':>7}"
      f"{'sigma_7/sigma_1':>18}{'SNR B=6':>14}")
print("-" * W)
for name in paths:
    for mode in rc_modes:
        r = pad_results[name][mode]
        print(f"{name:<11}{mode:<8}{r['R']:>5} x {r['C']:<3}{r['zeros']:>7}{r['taps']:>7}"
              f"{r['coefs']:>7}{r['sigma7']:>18.3e}{r['snr']:>11.2f} dB")
    print("-" * W)
print("=" * W)

# --- Stacked-matrix singular values for each mode ---
fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=[path_pt[name] for name in paths])
for col, name in enumerate(paths, 1):
    for j, (mode, dash) in enumerate(zip(rc_modes, ("solid", "dot", "dash"))):
        r = pad_results[name][mode]
        S_norm = r["S"][:12]
        color = px.colors.qualitative.Plotly[j]
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(S_norm) + 1), y=S_norm, mode="lines+markers",
            name=f"{mode} ({r['R']}×{r['C']}, {r['zeros']} zeros)",
            line=dict(dash=dash, color=color), marker=dict(color=color),
            legendgroup=mode, showlegend=(col == 1),
        ), row=1, col=col)
    fig.add_vline(x=B_test, line=dict(color="gray", dash="dot"), row=1, col=col)
fig.update_yaxes(type="log", range=[-17, 0.5])          # all subplots (shared axes do not copy the type)
fig.update_yaxes(title_text="σ<sub>k</sub> / σ<sub>1</sub>", row=1, col=1)
fig.update_xaxes(title_text="Índice k")
fig.update_layout(template="plotly_white", width=1000, height=450)
fig.show()
fig.write_image("figs/S_values_padding.pdf")


# %%
# =============================================================================
# EFFECT OF ALIASING ON THE RANK
# =============================================================================
# A mode above fs/2 is still a pair of complex exponentials, only with apparent
# frequency |f - k fs|. So aliasing does not ADD singular values beyond the pair
# of each mode: the rank is 2 * (number of modes in the sampled signal),
# including the modes above fs/2, not only the ones in the band of interest.
# Test: same beam, fs = 832 Hz (no aliasing for 3 modes) and fs = 416 Hz
# (mode 3 at ~250 Hz aliased to ~165 Hz), with 3 and 5 simulated modes.
# The rank is computed on the tail h[n_delay:] (see the delay section).
alias_cases = [(832.0, 3), (416.0, 3), (832.0, 5), (416.0, 5)]
alias_time  = 3.3                       # response length (s), ~settling time
alias_results = []

for fs_a, nm in alias_cases:
    beam_a = CantileverBeam(npoints=npoints, width=beamwidth, thickness=beamthickness,
                            length=beamlength, Tsampling=1.0 / fs_a,
                            damp=[0.01] * nm, nmodes=nm)
    f_modes  = beam_a.freqsHz[:nm]
    f_appar  = np.abs(f_modes - np.round(f_modes / fs_a) * fs_a)
    for name, out in (("Secondary", errorpos), ("Feedback", referencepos)):
        h_a  = get_impulse_response(beam_a, int(alias_time * fs_a), controlpos, out)
        tail = h_a[n_delay:]
        L_a  = int(np.sqrt(len(tail))) + 1
        S_a  = np.linalg.svd(lagged_matrix(tail, L_a), compute_uv=False)
        S_a  = S_a / S_a[0]
        alias_results.append(dict(fs=fs_a, nmodes=nm, path=name, f=f_modes,
                                  f_appar=f_appar, S=S_a,
                                  rank=int(np.sum(S_a > rank_tol))))

W = 108
print("\n" + "=" * W)
print(f"{'EFFECT OF ALIASING ON THE RANK (Hankel of h[%d:])' % n_delay:^{W}}")
print("=" * W)
print(f"{'fs (Hz)':>8}{'modes':>7}{'path':>11}  {'mode freqs (Hz)':<34}{'apparent (Hz)':<34}{'rank':>6}")
print("-" * W)
for r in alias_results:
    print(f"{r['fs']:>8.0f}{r['nmodes']:>7}{r['path']:>11}  "
          f"{np.array2string(r['f'], precision=1):<34}"
          f"{np.array2string(r['f_appar'], precision=1):<34}{r['rank']:>6}")
print("=" * W)

# --- Singular values: 832 Hz vs 416 Hz (3 and 5 modes, both paths) ---
fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=[path_pt[name] for name in paths])
styles = {(832.0, 3): ("#1D9E75", "solid"), (416.0, 3): ("#E8833A", "dash"),
          (832.0, 5): ("#5B8DEF", "solid"), (416.0, 5): ("#E84393", "dash")}
for col, name in enumerate(paths, 1):
    for r in (r for r in alias_results if r["path"] == name):
        color, dash = styles[(r["fs"], r["nmodes"])]
        S_norm = r["S"][:14]
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(S_norm) + 1), y=S_norm, mode="lines+markers",
            name=f"f<sub>s</sub> = {r['fs']:.0f} Hz, {r['nmodes']} modos",
            line=dict(color=color, dash=dash), marker=dict(color=color),
            legendgroup=f"{r['fs']}-{r['nmodes']}", showlegend=(col == 1),
        ), row=1, col=col)
fig.update_yaxes(type="log", range=[-17, 0.5])
fig.update_yaxes(title_text="σ<sub>k</sub> / σ<sub>1</sub>", row=1, col=1)
fig.update_xaxes(title_text="Índice k")
fig.update_layout(template="plotly_white", width=1000, height=450)
fig.show()
fig.write_image("figs/S_values_aliasing.pdf")

# --- Coincident poles: when aliasing DOES reduce the rank ---
# The rank is 2 per mode only while the poles z = exp((-beta + j w) / fs) are
# distinct. A mode at f_b = f_a + fs with the SAME decay rate beta = zeta*w_n
# has exactly the same z as the mode at f_a (aliased onto it), so the two
# modes merge into a single pair and the rank drops from 4 to 2.
# Synthetic test (not the beam): two decaying cosines, fs = 832 Hz.
fs_c, beta_c, f_a = 832.0, 2.0, 89.5          # beta in 1/s
n_c = np.arange(int(3.3 * fs_c))
coincident = []
for label, f_b in (("f_b = f_a + 37 Hz (distinct z)", f_a + 37.0),
                   ("f_b = f_a + fs (same z)",        f_a + fs_c)):
    h_c = (np.exp(-beta_c * n_c / fs_c) * np.cos(2 * np.pi * f_a * n_c / fs_c)
           + 0.5 * np.exp(-beta_c * n_c / fs_c) * np.cos(2 * np.pi * f_b * n_c / fs_c + 0.3))
    L_c = int(np.sqrt(len(h_c))) + 1
    S_c = np.linalg.svd(lagged_matrix(h_c, L_c), compute_uv=False)
    coincident.append((label, int(np.sum(S_c / S_c[0] > rank_tol))))
print("\nCoincident-pole test (2 modes, same decay rate):")
for label, rk in coincident:
    print(f"  {label:<34} rank = {rk}")


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

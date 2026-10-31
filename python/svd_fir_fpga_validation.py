# %%
# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import chirp

from CantileverBeam import CantileverBeam
from aux_functions import (
    norm, format_W, lagged_matrix, svd_branch_weights, gen_SVD_weights,
    FIRSVDFilterPy, gen_wsec_wfbk_filters, quantize_matrix, weights_to_hex,
    matrix_to_sv_hex, write_hex, plot_fft_onesided, plot_phase_onesided,
    plot_paired_branches_fft, plot_cumulative_branches_fft,
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
    title="Cantilever Beam",
    xaxis_title="x (m)", yaxis_title="y (m)",
    xaxis=dict(range=[0, beamlength * 1.1]),
    yaxis=dict(range=[-(beamthickness + 0.1), beamthickness + 0.1]),
    width=600, height=350
)
fig.show()


# %%
# =============================================================================
# OPEN-LOOP VIBRATION SIMULATION
# =============================================================================
# Simulates the beam response to a sinusoidal perturbation force.
# The force starts at vibstart seconds and runs until maxtime.
# Plots both the input force and the resulting acceleration response.

maxtime  = 60.0
vibstart = 10
vibfreq  = 80.0   # Hz

nsteps = int(maxtime * fs)
th = np.linspace(0.0, maxtime, nsteps)
xh = 0.3 * np.sin(2 * np.pi * th * vibfreq)
xh[0:int(fs * vibstart)] = 0.0   # zero force before vibstart

cbeam.reset()
err = np.zeros(nsteps)
for k in range(nsteps):
    err[k] = cbeam.getaccelms2(referencepos)
    cbeam.setforce(controlpos, xh[k])
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
print(wsecimpulse.shape, wfbkimpulse.shape)

#filter_size = 6400
filter_size = 2 * (64 + 101)
# always_pad=True keeps R = 101 (6400 taps / 64 + 1), as expected by svd_filter_tb.sv
C_weights_sec, R_weights_sec , S_emb= gen_SVD_weights(wsecimpulse, filter_size, C_size=64, plot_S=True,
                                                      always_pad=True)
print(f"Vt shape: {C_weights_sec.shape}  Us shape: {R_weights_sec.shape}")

# Quantize with a shared scale so relative magnitudes are preserved
max_ov = np.max([np.max(np.abs(C_weights_sec)), np.max(np.abs(R_weights_sec))])

Vt_q   = quantize_matrix(C_weights_sec, np.max(np.abs(C_weights_sec)))
Us_q   = quantize_matrix(R_weights_sec, np.max(np.abs(R_weights_sec)))

plot_fft_onesided(wsecimpulse, fs ,"Secondary Path FFT")
plot_fft_onesided(wfbkimpulse, fs ,"Feedback Path FFT")


fig = px.line()
fig.add_scatter(y=Vt_q[0, :], name="Vt b=0")
fig.add_scatter(y=Us_q[0, :], name="Us b=0")
fig.update_layout(title="Quantized SVD Weights — Branch 0")
fig.show()

# Print as SystemVerilog parameters (for reference / small filters)
matrix_to_sv_hex(Vt_q, "VT")
matrix_to_sv_hex(Us_q, "US")

# %%
# =============================================================================
# PYTHON REFERENCE SIMULATION — IMPULSE RESPONSE
# =============================================================================
# Runs the Python SVD filter with a unit impulse input to obtain the
# expected impulse response. Used to validate the FPGA output via FFT comparison.

svd_filter_py = FIRSVDFilterPy(C_weights_sec, R_weights_sec)
svd_filter_py.reset()

x = np.zeros(1000)
x[0] = 1.0
y = np.zeros_like(x)
for i, sample in enumerate(x):
    y[i] = svd_filter_py.filterstep(sample)

SVD_py_output = y

fig = px.line()
fig.add_scatter(y=x, name="Input (impulse)")
fig.add_scatter(y=y, name="Output y")
fig.update_layout(title="Python SVD Filter — Impulse Response")
fig.show()

plot_fft_onesided(y, fs=400, title="FFT — Python SVD Filter Output")

# %%
# =============================================================================
# INPUT SAMPLE HEX FILE GENERATION
# =============================================================================
# Quantizes the input sequence to 8-bit signed and writes samples.hex
# for use with $readmemh in the testbench.

fs = 400                    # Sampling frequency (Hz)
N = len(x)                  # Number of samples
t = np.arange(N) / fs


#IMPULSE SIGNAL
impulse = np.zeros(N)
impulse[0] = 1.0

svd_filter_py.reset()
for i, sample in enumerate(impulse):
    y[i] = svd_filter_py.filterstep(sample)
impulse_py_output = np.copy(y)

write_hex(impulse, "impulse.hex")

#HARMONIC 86 Hz SIGNAL
f0 = 86  # Hz
harmonic = np.sin(2 * np.pi * f0 * t)

svd_filter_py.reset()
for i, sample in enumerate(harmonic):
    y[i] = svd_filter_py.filterstep(sample)
harmonic_py_output = np.copy(y)
write_hex(harmonic, "harmonic.hex")

#DECREASING HARMONIC 86 Hz SIGNAL
amp = np.arange(0,1,1/N)

svd_filter_py.reset()
for i, sample in enumerate(harmonic*amp):
    y[i] = svd_filter_py.filterstep(sample)
harmonic_dec_py_output = np.copy(y)
write_hex(harmonic*amp, "harmonic_dec.hex")

#CHIRP SIGNAL

s_chirp = chirp(
    t,
    f0=0,
    t1=t[-1],
    f1=180,
    method='linear'
)
svd_filter_py.reset()
for i, sample in enumerate(s_chirp):
    y[i] = svd_filter_py.filterstep(sample)
chirp_py_output = np.copy(y)
write_hex(s_chirp, "chirp.hex")

fig = px.line()
fig.add_scatter(y=impulse, name="Impulse")
fig.add_scatter(y=harmonic, name="Harmonic")
fig.add_scatter(y=s_chirp, name="Chirp")
fig.add_scatter(y=harmonic*amp, name="Harmonic Decreasing")
fig.update_layout(title="Test Signals")
fig.show()

# %%
# =============================================================================
# WEIGHT HEX FILE GENERATION — SVD BRANCHES
# =============================================================================
# Writes per-branch Vt and Us weight files as .hex for $readmemh.
# File naming: B<branch><VT/US>_weights.hex

print(f"Current working directory: {os.getcwd()}")

weights_to_hex(Vt_q[0, :], "../weights/B0VT_weights.hex", width=8, quant=True)
weights_to_hex(Vt_q[1, :], "../weights/B1VT_weights.hex", width=8, quant=True)
weights_to_hex(Us_q[0, :], "../weights/B0US_weights.hex", width=8, quant=True)
weights_to_hex(Us_q[1, :], "../weights/B1US_weights.hex", width=8, quant=True)

# %%
# =============================================================================
# WEIGHT HEX FILE GENERATION — PLAIN FIR (BASELINE)
# =============================================================================
# Generates the full FIR weight file for the baseline (non-SVD) FPGA filter.
# Used to compare area/performance against the SVD implementation.

fir_weights = wsecimpulse
max_ov      = np.max(np.abs(fir_weights))
FIR_w       = quantize_matrix(fir_weights, max_ov, width=8)

print(f"FIR weights shape: {FIR_w.shape}  range: [{FIR_w.min()}, {FIR_w.max()}]")

fig = px.line()
fig.add_scatter(y=FIR_w, name="FIR weights (quantized)")
fig.update_layout(title="Quantized FIR Weights")
fig.show()

matrix_to_sv_hex(FIR_w, "W")
weights_to_hex(FIR_w, "../weights/FIR_weights.hex", width=16)

# %%
# =============================================================================
# FPGA OUTPUT COMPARISON
# =============================================================================
# Loads results.txt written by the SystemVerilog testbench and compares
# the FPGA output against the Python reference via time-domain and FFT plots.

df_svd = pd.read_csv("../work/SVD_results_impulse.txt", skipinitialspace=True)
df_fir = pd.read_csv("../work/FIR_results.txt", skipinitialspace=True)

fig = px.line()
fig.add_scatter(x=df_svd['idx'],y=df_svd['y'], name="y SVD FPGA")
fig.update_layout(title="FPGA Output")
fig.show()

plot_fft_onesided(df_svd['y'], fs=400, title="FFT — SVD FPGA Output")
plot_fft_onesided(y,       fs=400, title="FFT — Python Reference")

fig = px.line()

fig.add_scatter(x=df_svd['idx'],y=norm(FIR_w),   name="FIR weights (Python)",  line=dict(width=5))
fig.add_scatter(x=df_svd['idx'],y=norm(df_svd['y']), name="SVD FPGA output")
fig.add_scatter(x=df_svd['idx'],y=norm(SVD_py_output), name="SVD weights (Python)")

fig.add_scatter(x=df_svd['idx'],y = norm(df_svd['y']) - norm(SVD_py_output), name= "Error SVD (FPGA - Python)")

fig.update_layout(title="Python vs FPGA — Normalized Comparison")
fig.show()

print (f"Number of cycles Compression \n")
print (f"SVD num cycles per operation: {df_svd["cycle_cnt"][0]}")
print (f"FIR num cycles per operation: {df_fir["cycle_cnt"][0]}")
print (f"Num cycles per operation compression: {100*df_fir["cycle_cnt"][0]/df_svd["cycle_cnt"][0]}x")
print("-------------------------------------------------------------------------------")
print("This SVD is composed of 2 branches, with 2 DSPs per branch.\n" \
"The number of cycles for a FIR filter with four DSPs would be\naround 4x less than" \
" the simulated one with one DSP only. ")

print(f"Num cycles per operation compression FIR w/4DSP: {100*df_fir["cycle_cnt"][0]/(4*df_svd["cycle_cnt"][0])}x")
#%%
# =============================================================================
# FPGA OUTPUT COMPARISON
# =============================================================================
# Compare IMPULSE, HARMONIC and CHIRP TESTS (Fine tune for GUARD and offset)
df_imp = pd.read_csv("../work/SVD_results_impulse.txt", skipinitialspace=True)
df_harmonic = pd.read_csv("../work/SVD_results_harmonic.txt", skipinitialspace=True)
df_chirp = pd.read_csv("../work/SVD_results_chirp.txt", skipinitialspace=True)
df_harmonic_dec = pd.read_csv("../work/SVD_results_harmonic_dec.txt", skipinitialspace=True)
fig = px.line()
#For impulse responses to be similar, guard must be zero,
fig.add_scatter(x=df_imp['idx'],y=(df_imp['y']),   name="SVD impulse response")
fig.add_scatter(x=df_imp['idx'],y=(impulse_py_output),   name="SVD impulse ouput py")

fig.add_scatter(x=df_harmonic['idx'],y=(df_chirp['y']), name="SVD Chirp response")
fig.add_scatter(x=df_harmonic['idx'],y=(chirp_py_output), name="SVD Chirp output py")

fig.add_scatter(x=df_harmonic['idx'],y=(harmonic_py_output), name="SVD Harmonic output py")
fig.add_scatter(x=df_harmonic['idx'],y=(df_harmonic['y']), name="SVD Harmonic response")

fig.add_scatter(x=df_harmonic_dec['idx'],y=(harmonic_dec_py_output), name="SVD Dec Harmonic output py")
fig.add_scatter(x=df_harmonic_dec['idx'],y=(df_harmonic_dec['y']), name="SVD Dec Harmonic response")

fig.update_layout(title="FPGA tests outputs")
fig.show()

# %%
plot_fft_onesided(df_harmonic['y'], fs=400, title="FFT — SVD FPGA Output Harmonic")

# %%
# Run fft for individual branches, check if 2 branch per mode is visible

svd_outs = np.array(svd_filter_py.youts_outs)

plot_fft_onesided(y, fs=400, title=f"FFT — Python SVD Filter Output")


for i in range(svd_outs.shape[1]) :
    plot_fft_onesided(svd_outs[:,i], fs=400, title=f"FFT — Python SVD Filter Output Branch {i}")


# %%
# =============================================================================
# SVD SPECTRUM ANALYSIS USING LAGGED MATRICE
# =============================================================================
#Compress filter to around 2400 taps
mem = 2400
wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam, mem, controlpos, errorpos, referencepos)
plot_fft_onesided(wsecimpulse , fs= 400, title="FFT secondary path")
# Create lagged matrice
L = 1200
K = mem - L + 1
W   = format_W(wsecimpulse, L)
print(f'{W.shape = }')

wsec_lagged = lagged_matrix(wsecimpulse, L)
#Make SVD of lagged vector
U, S_lagged, VT = np.linalg.svd(wsec_lagged)
print(U.shape , S_lagged.shape , VT.shape)
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=S_lagged/np.max(S_lagged),
    mode="lines",
    name="Lagged"
))
fig.add_trace(go.Scatter(
    y=S_emb/np.max(S_emb),
    mode="lines",
    name="Stacked"
))
fig.update_layout(
        title="Normalized EigenValues of SVD on Lagged and Stacked matrice"
)
fig.show()
# %%

B_s = 6
C_weights, R_weights = svd_branch_weights(U, S_lagged, VT, B_s)

svd_lagged = FIRSVDFilterPy(C_weights, R_weights)
svd_lagged.reset()

x = np.zeros(1000)
x[0] = 1.0
y = np.zeros_like(x)
for i, sample in enumerate(x):
    y[i] = svd_lagged.filterstep(sample)

svd_lagged_outs = np.array(svd_lagged.youts_outs)

# %%
plot_fft_onesided(y, fs=fs, title=f"FFT — Python SVD Filter Output")
for i in range(svd_lagged_outs.shape[1]) :
    plot_fft_onesided(svd_lagged_outs[:,i], fs=fs, title=f"FFT — Python SVD Filter Output Branch {i}")

#%%
fig = plot_paired_branches_fft(svd_lagged_outs, fs=fs, ylim=(-100, 0))
fig.write_image("figs/Hankel_Branches_FFT.pdf") 
#%%
#Generate Same Frequency Response for feedback implementation
W   = format_W(wfbkimpulse, L)
print(f'{W.shape = }')

wfbk_lagged = lagged_matrix(wfbkimpulse, L)
#Make SVD of lagged vector
U, S_lagged, VT = np.linalg.svd(wfbk_lagged)

B_s = 6
C_weights, R_weights = svd_branch_weights(U, S_lagged, VT, B_s)

svd_lagged = FIRSVDFilterPy(C_weights, R_weights)
svd_lagged.reset()

x = np.zeros(1000)
x[0] = 1.0
y = np.zeros_like(x)
for i, sample in enumerate(x):
    y[i] = svd_lagged.filterstep(sample)

svd_lagged_outs_fbk = np.array(svd_lagged.youts_outs)


fig = plot_paired_branches_fft(svd_lagged_outs_fbk, fs=fs, ylim=(-100, -10))
fig.write_image("figs/Hankel_Branches_FFT_fbk.pdf") 
#%%

plot_fft_onesided(y,fs,"FFT of whole feedback")
plot_phase_onesided(y,fs,"FFT phase of whole feedback")


#%%
#Generate Same Frequency Response for feedback implementation
filter_size = 6 * (64 + 101)
C_weights_fbk, R_weights_fbk , S_emb= gen_SVD_weights(wfbkimpulse, filter_size, C_size=64, plot_S=True)

svd_stack = FIRSVDFilterPy(C_weights_fbk, R_weights_fbk)
svd_stack.reset()

x = np.zeros(1000)
x[0] = 1.0
y = np.zeros_like(x)
for i, sample in enumerate(x):
    y[i] = svd_stack.filterstep(sample)

svd_stack_outs_fbk = np.array(svd_stack.youts_outs)


fig = plot_paired_branches_fft(svd_stack_outs_fbk, fs=fs, ylim=(-100, -10),group_size  = 1)
fig.write_image("figs/Hankel_Branches_FFT_fbk_stack.pdf") 
plot_fft_onesided(y , fs , "FFT- SVD 6 Branches")

# %%
plot_cumulative_branches_fft(svd_stack_outs_fbk, fs=fs)           # 1, 1+2, 1+2+3, ...
# %%

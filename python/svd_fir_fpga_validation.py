# %%
# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

from CantileverBeam import CantileverBeam
from Adaptive import FIRNLMS
from AdaptiveOO import FIRFxNLMS, FIR

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
    damp=dampingfactors
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
# HELPER FUNCTIONS
# =============================================================================
# Utility functions used throughout the script:
#   format_W          : reshapes and zero-pads a weight vector into a 2D matrix
#   gen_wsec_wfbk_filters : computes impulse responses for secondary/feedback paths
#   gen_SVD_weights   : performs SVD decomposition and extracts branch weights
#   quantize_matrix   : quantizes float matrix to signed fixed-point integers
#   weights_to_hex    : writes a numpy array to a .hex file for $readmemh
#   matrix_to_sv_hex  : prints weights as SystemVerilog parameter hex literals
#   plot_fft_onesided : plots one-sided FFT magnitude spectrum using Plotly
#   norm              : normalizes array to [-1, 1]

def format_W(W, C):
    """Zero-pad and reshape weight vector into a 2D matrix of C columns."""
    n_pad = W.shape[0] % C
    W_pad = np.zeros(int(W.shape[0] + (C - n_pad)))
    W_pad[:W.shape[0]] = W
    R = W_pad.shape[0] / C
    lower_dim = min(R, C)
    return W_pad.reshape(-1, int(lower_dim), order='F')


def gen_wsec_wfbk_filters(firmem):
    """
    Compute impulse responses for the secondary path (control→error)
    and feedback path (control→reference) by simulating the beam.
    Returns (wsecimpulse, wfbkimpulse).
    """
    def get_impulse(output_pos):
        impulse = np.zeros(firmem)
        cbeam.reset()
        cbeam.setforce(controlpos, 1.0)
        cbeam.update()
        impulse[0] = cbeam.getaccelms2(output_pos)
        cbeam.setforce(controlpos, 0.0)
        for k in range(1, firmem):
            cbeam.update()
            impulse[k] = cbeam.getaccelms2(output_pos)
        return impulse

    wsecimpulse = get_impulse(errorpos)
    wfbkimpulse = get_impulse(referencepos)

    fig = px.line()
    fig.add_scatter(y=wfbkimpulse, name="Feedback path")
    fig.add_scatter(y=wsecimpulse, name="Secondary path")
    fig.show()

    return wsecimpulse, wfbkimpulse


def gen_SVD_weights(weights, num_coef, C_size=0):
    """
    Decompose a FIR filter weight vector using SVD and extract
    branch weights (C_weights, R_weights) for a given number of
    equivalent coefficients.

    Args:
        weights  : 1D array of FIR filter coefficients
        num_coef : target number of coefficients after SVD compression
        C_size   : number of columns in the reshaped matrix (0 = auto sqrt)

    Returns:
        C_weights : (B, C) right-singular vectors (Vt rows)
        R_weights : (B, R) left-singular * sigma vectors (Us rows)
    """
    full_size  = len(weights)
    C_chosen_s = C_size if C_size != 0 else int(np.sqrt(full_size))
    W          = format_W(weights, C_chosen_s)
    print(f'{W.shape = }')
    R_s = W.shape[0]
    B_s = int(np.round(num_coef / (C_chosen_s + R_s)))

    U, S, VT = np.linalg.svd(W)
    SM = np.zeros((R_s, C_chosen_s))
    np.fill_diagonal(SM, S)
    US = U @ SM

    C_weights = np.zeros((B_s, VT.shape[1]))
    R_weights = np.zeros((B_s, U.shape[0]))
    for i in range(B_s):
        C_weights[i, :] = VT.T[:, i]
        R_weights[i, :] = US[:, i]

    return C_weights, R_weights


def quantize_matrix(matrix, max_ov, width=8):
    """
    Quantize a float matrix to signed fixed-point integers.
    Uses a global max_ov for consistent scaling across matrices.
    """
    scale = (2 ** (width - 1) - 1) / max_ov if max_ov != 0 else 1
    quantized = np.clip(
        np.round(matrix * scale),
        -(2 ** (width - 1)),
        2 ** (width - 1) - 1
    )
    return quantized.astype(np.int8)


def weights_to_hex(weights: np.ndarray, filename: str,
                   width: int = 8, quant: bool = False) -> None:
    """
    Write a numpy array of filter weights to a .hex file for $readmemh.
    If quant=False, quantizes from float. If quant=True, uses values as-is.

    Args:
        weights  : numpy array (any shape, will be flattened)
        filename : output .hex file path
        width    : bit width per coefficient (default 8)
        quant    : if True, skip quantization (array already quantized)
    """
    if not quant:
        max_val   = np.max(np.abs(weights))
        scale     = (2 ** (width - 1) - 1) / max_val if max_val != 0 else 1.0
        quantized = np.clip(
            np.round(weights.flatten() * scale),
            -(2 ** (width - 1)),
            (2 ** (width - 1)) - 1
        ).astype(np.int8 if width <= 8 else np.int16)
    else:
        quantized = weights.flatten()

    hex_digits = width // 4
    with open(filename, "w") as f:
        for v in quantized:
            f.write(f"{int(v) & (2 ** width - 1):0{hex_digits}X}\n")

    print(f"Written {filename} — {len(quantized)} entries  "
          f"range=[{quantized.min()}, {quantized.max()}]")


def matrix_to_sv_hex(matrix, name, width=16):
    """
    Print a matrix as SystemVerilog parameter hex literals.
    Columns are reversed so that flat[i*WIDTH +: WIDTH] == col[i].
    """
    if len(matrix.shape) == 2:
        rows, cols = matrix.shape
    else:
        cols, rows = len(matrix), 1

    hex_digits = width // 4
    print(f"// {name} — shape ({rows}, {cols})")
    for r in range(rows):
        row_data = matrix[r] if rows != 1 else matrix
        row_hex  = "".join(
            f"{int(v) & (2 ** width - 1):0{hex_digits}X}"
            for v in reversed(row_data)
        )
        total_bits = cols * width
        print(f"// ({r}, :)")
        print(f"parameter [{total_bits - 1}:0] {name}_ROW{r} = {total_bits}'h{row_hex};")
    print()


def plot_fft_onesided(sig, fs=1.0, title="FFT"):
    """Plot one-sided FFT magnitude spectrum in dB using Plotly."""
    N      = len(sig)
    sig_n  = sig / np.max(np.abs(sig))
    fft_mag = np.abs(np.fft.rfft(sig_n)) / N
    fft_mag[1:-1] *= 2
    fft_db  = 20 * np.log10(fft_mag + 1e-12)
    freqs   = np.fft.rfftfreq(N, d=1 / fs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=fft_db, mode="lines",
        line=dict(color="#1D9E75", width=1.5)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_dark",
        hovermode="x unified"
    )
    fig.show()


def norm(x):
    """Normalize array to [-1, 1]."""
    return x / np.max(np.abs(x))

# %%
# =============================================================================
# SVD FILTER — PYTHON REFERENCE IMPLEMENTATION
# =============================================================================
# Implements the SVD-based FIR filter in pure Python.
# Used as the golden reference to compare against the FPGA implementation.
# Architecture:
#   1. Input buffer of length N = R*C (circular)
#   2. Vt (C_weights) projects the downsampled buffer onto B scalars
#   3. Each scalar is filtered by an FIR with R_weights[b]
#   4. Outputs are summed to produce y

class FIRSVDFilterPy(FIR):
    def __init__(self, C_weights, R_weights):
        self.R   = R_weights.shape[1]
        self.C   = C_weights.shape[1]
        self.B   = C_weights.shape[0]
        self.N   = self.R * self.C
        self.vdot = C_weights
        self.inputbuffer = np.zeros(self.N)
        self.util = [FIR(R_weights[k, :]) for k in range(self.B)]
        self.reset()

    def reset(self):
        self.y = 0
        self.inputbuffer[:] = 0.0
        for k in range(self.B):
            self.util[k].reset()

    def filterstep(self, xn):
        self.inputbuffer[1:] = self.inputbuffer[:-1]
        self.inputbuffer[0]  = xn
        self.youts = self.vdot @ self.inputbuffer[::self.R]
        for k in range(self.B):
            self.youts[k] = self.util[k].filterstep(self.youts[k])
        self.y = np.sum(self.youts)
        return self.y

# %%
# =============================================================================
# IMPULSE RESPONSE & SVD WEIGHT GENERATION
# =============================================================================
# Computes the secondary path impulse response, then runs SVD decomposition
# to obtain Vt (C_weights) and Us (R_weights) for each branch.
# Also quantizes the weights and plots them for inspection.

wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(firmem)
print(wsecimpulse.shape, wfbkimpulse.shape)

filter_size = 256
C_weights_sec, R_weights_sec = gen_SVD_weights(wsecimpulse, filter_size, C_size=64)
print(f"Vt shape: {C_weights_sec.shape}  Us shape: {R_weights_sec.shape}")

# Quantize with a shared scale so relative magnitudes are preserved
max_ov = np.max([np.max(np.abs(C_weights_sec)), np.max(np.abs(R_weights_sec))])
Vt_q   = quantize_matrix(C_weights_sec, max_ov)
Us_q   = quantize_matrix(R_weights_sec, max_ov)

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

SCALE     = 127.0 / np.max(np.abs(x))
quantized = np.clip(np.round(x * SCALE), -128, 127).astype(np.int8)

with open("samples.hex", "w") as f:
    for s in quantized:
        f.write(f"{int(s) & 0xFF:02X}\n")

print(f"Written samples.hex — {len(quantized)} entries")

# %%
# =============================================================================
# WEIGHT HEX FILE GENERATION — SVD BRANCHES
# =============================================================================
# Writes per-branch Vt and Us weight files as .hex for $readmemh.
# File naming: B<branch><VT/US>_weights.hex

print(f"Current working directory: {os.getcwd()}")

weights_to_hex(Vt_q[0, :], "../weights/B0VT_weights.hex", width=16, quant=True)
weights_to_hex(Vt_q[1, :], "../weights/B1VT_weights.hex", width=16, quant=True)
weights_to_hex(Us_q[0, :], "../weights/B0US_weights.hex", width=16, quant=True)
weights_to_hex(Us_q[1, :], "../weights/B1US_weights.hex", width=16, quant=True)

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

df_svd = pd.read_csv("../work/SVD_results.txt", skipinitialspace=True)
df_fir = pd.read_csv("../work/FIR_results.txt", skipinitialspace=True)

fig = px.line()
fig.add_scatter(y=df_svd['y'], name="y SVD FPGA")
fig.update_layout(title="FPGA Output")
fig.show()

plot_fft_onesided(df_svd['y'], fs=400, title="FFT — SVD FPGA Output")
plot_fft_onesided(y,       fs=400, title="FFT — Python Reference")

fig = px.line()

fig.add_scatter(y=norm(FIR_w),   name="FIR weights (Python)",  line=dict(width=5))
fig.add_scatter(y=norm(df_svd['y']), name="SVD FPGA output")
fig.add_scatter(y=norm(SVD_py_output), name="SVD weights (Python)")

fig.add_scatter(y = norm(df_svd['y']) - norm(SVD_py_output), name= "Error SVD (FPGA - Python)")

fig.update_layout(title="Python vs FPGA — Normalized Comparison")
fig.show()



# %%

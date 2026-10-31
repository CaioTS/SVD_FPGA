# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================
# Shared helpers used by the analysis / validation scripts in this folder.
#
# Matrix & SVD
#   norm                      : normalizes array to [-1, 1]
#   format_W                  : reshapes and zero-pads a weight vector into a 2D matrix
#   lagged_matrix             : builds the Hankel (lagged) matrix of a vector
#   svd_report                : SVD of a matrix with a one-line shape summary
#   svd_branch_weights        : extracts (C_weights, R_weights) from an SVD
#   gen_SVD_weights           : SVD branch weights for a target number of coefficients
#   gen_SVD_weights_branches  : SVD branch weights for a given number of branches
#   FIRSVDFilterPy            : Python reference implementation of the SVD FIR filter
#   run_filter                : runs a filter over a whole signal
#
# Beam
#   get_impulse_response      : impulse response between two FE nodes of the beam
#   gen_wsec_wfbk_filters     : secondary and feedback path impulse responses
#
# Hardware export
#   quantize_matrix           : quantizes float matrix to signed fixed-point integers
#   weights_to_hex            : writes a numpy array to a .hex file for $readmemh
#   matrix_to_sv_hex          : prints weights as SystemVerilog parameter hex literals
#   write_hex                 : writes an 8-bit quantized input signal to a .hex file
#
# Spectrum & plots
#   onesided_db               : one-sided FFT magnitude in dB
#   plot_fft_onesided         : plots the FFT magnitude of one signal
#   plot_fft_onesided_pair    : plots the FFT magnitude of two signals
#   add_fft_trace             : adds an FFT magnitude trace to an existing figure
#   plot_phase_onesided       : plots the unwrapped FFT phase of one signal
#   interpolate_x             : linear interpolation of x for a target y
#   plot_fft_onesided_n_modes : FFT with resonance and -3 dB detection
#   plot_singular_values      : compares two singular value spectra
#   plot_paired_branches_fft  : FFT of branch outputs summed in groups
#   plot_cumulative_branches_fft : FFT of cumulatively summed branch outputs
#   compare_filters           : impulse response error between SVD FIR and FIR
#
# Resource utilization
#   calc_svd_utilizations     : SVD FIR resources from its weight matrices
#   calc_svd_utilizations_teo : SVD FIR resources from filter size and branches
#   calc_fir_utilization      : conventional FIR resources from filter size

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

from AdaptiveOO import FIR


# =============================================================================
# MATRIX & SVD
# =============================================================================

def norm(x):
    """Normalize array to [-1, 1]."""
    return x / np.max(np.abs(x))


def format_W(W, C, always_pad=False):
    """
    Zero-pad and reshape weight vector into a 2D matrix of C columns.

    If always_pad is True, a full extra block of C zeros is appended even when
    len(W) is already a multiple of C (original behaviour, e.g. 6400 taps with
    C=64 gives R=101, which the FPGA testbench expects).
    """
    n_pad = W.shape[0] % C
    if n_pad == 0 and not always_pad:
        n_pad = C
    W_pad = np.zeros(int(W.shape[0] + (C - n_pad)))
    W_pad[:W.shape[0]] = W
    R = W_pad.shape[0] / C
    lower_dim = min(R, C)
    return W_pad.reshape(-1, int(lower_dim), order='F')


def lagged_matrix(x, L):
    """Hankel matrix H[l, k] = x[l + k], shape (L, len(x) - L + 1)."""
    K = len(x) - L + 1
    return sliding_window_view(x, K)[:L]


def svd_report(name, A):
    """Compute the SVD of A and print a one-line shape summary."""
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    print(f"{name:<8} A: {A.shape} | U: {U.shape} | S: {S.shape} | VT: {VT.shape}")
    return U, S, VT


def svd_branch_weights(U, S, VT, n_branches):
    """
    Build the branch weights of the SVD filter from a full SVD.

    Returns:
        C_weights : (B, C) right-singular vectors (Vt rows)
        R_weights : (B, R) left-singular * sigma vectors (Us rows)
    """
    SM = np.zeros((U.shape[0], VT.shape[0]))
    np.fill_diagonal(SM, S)
    US = U @ SM

    C_weights = np.zeros((n_branches, VT.shape[1]))
    R_weights = np.zeros((n_branches, U.shape[0]))
    for i in range(n_branches):
        C_weights[i, :] = VT.T[:, i]
        R_weights[i, :] = US[:, i]
    return C_weights, R_weights


def gen_SVD_weights_branches(weights, n_branches, C_size=0, plot_S=False,
                             always_pad=False, verbose=True):
    """
    Decompose a FIR filter weight vector using SVD and extract
    branch weights (C_weights, R_weights) for a given number of branches.

    Args:
        weights    : 1D array of FIR filter coefficients
        n_branches : number of SVD branches kept
        C_size     : number of columns in the reshaped matrix (0 = auto sqrt)
        always_pad : passed to format_W
        verbose    : print the shape of the reshaped matrix

    Returns:
        C_weights : (B, C) right-singular vectors (Vt rows)
        R_weights : (B, R) left-singular * sigma vectors (Us rows)
        S         : all singular values
    """
    full_size  = len(weights)
    C_chosen_s = C_size if C_size != 0 else int(np.sqrt(full_size))
    W          = format_W(weights, C_chosen_s, always_pad)
    if verbose:
        print(f'{W.shape = }')

    U, S, VT = np.linalg.svd(W)
    C_weights, R_weights = svd_branch_weights(U, S, VT, n_branches)

    if (plot_S):
        fig = px.line()
        fig.add_scatter(y=S/np.max(S))
        fig.update_layout(title="Singular Values")

        fig.show()
    return C_weights, R_weights , S


def gen_SVD_weights(weights, num_coef, C_size=0, plot_S=False, always_pad=False,
                    verbose=True):
    """
    Same as gen_SVD_weights_branches, but the number of branches is derived
    from num_coef, the target number of coefficients after SVD compression:
    B = round(num_coef / (C + R)).
    """
    C_chosen_s = C_size if C_size != 0 else int(np.sqrt(len(weights)))
    R_s = format_W(weights, C_chosen_s, always_pad).shape[0]
    n_branches = int(np.round(num_coef / (C_chosen_s + R_s)))
    return gen_SVD_weights_branches(weights, n_branches, C_chosen_s, plot_S,
                                    always_pad, verbose)


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
        self.youts_outs = []
        self.reset()

    def reset(self):
        self.y = 0
        self.inputbuffer[:] = 0.0
        self.youts_outs = []
        for k in range(self.B):
            self.util[k].reset()

    def filterstep(self, xn):
        self.inputbuffer[1:] = self.inputbuffer[:-1]
        self.inputbuffer[0]  = xn
        self.youts = self.vdot @ self.inputbuffer[::self.R]
        for k in range(self.B):
            self.youts[k] = self.util[k].filterstep(self.youts[k])
        self.y = np.sum(self.youts)
        self.youts_outs.append(self.youts)
        return self.y


def run_filter(flt, x):
    """Run a filter over x. Tries the whole array first, falls back to sample by sample."""
    try:
        y = flt.filter(x)
        y = np.asarray(y).ravel()
        if len(y) == len(x):
            return y
    except (TypeError, ValueError):
        pass
    return np.array([np.asarray(flt.filter(xi)).item() for xi in x])


# =============================================================================
# BEAM
# =============================================================================

def get_impulse_response(cbeam, firmem, input_pos, output_pos):
    """
    Impulse response of the beam from a force at FE node input_pos to the
    acceleration at FE node output_pos, firmem samples long.
    """
    impulse = np.zeros(firmem)
    cbeam.reset()
    cbeam.setforce(input_pos, 1.0)
    cbeam.update()
    impulse[0] = cbeam.getaccelms2(output_pos)
    cbeam.setforce(input_pos, 0.0)
    for k in range(1, firmem):
        cbeam.update()
        impulse[k] = cbeam.getaccelms2(output_pos)
    return impulse


def gen_wsec_wfbk_filters(cbeam, firmem, controlpos, errorpos, referencepos):
    """
    Compute impulse responses for the secondary path (control→error)
    and feedback path (control→reference) by simulating the beam.
    Returns (wsecimpulse, wfbkimpulse).
    """
    wsecimpulse = get_impulse_response(cbeam, firmem, controlpos, errorpos)
    wfbkimpulse = get_impulse_response(cbeam, firmem, controlpos, referencepos)

    fig = px.line()
    fig.add_scatter(y=wfbkimpulse, name="Feedback path")
    fig.add_scatter(y=wsecimpulse, name="Secondary path")
    fig.show()

    return wsecimpulse, wfbkimpulse


# =============================================================================
# HARDWARE EXPORT
# =============================================================================

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


def matrix_to_sv_hex(matrix, name, width=8):
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


def write_hex(signal, filename):
    """Quantize a signal to 8-bit signed and write it to a .hex file for $readmemh."""
    scale = 127.0 / np.max(np.abs(signal))
    quantized = np.clip(np.round(signal * scale), -128, 127).astype(np.int8)

    with open(filename, "w") as f:
        for s in quantized:
            f.write(f"{int(s) & 0xFF:02X}\n")

    print(f"Written {filename} — {len(signal)} entries")


# =============================================================================
# SPECTRUM & PLOTS
# =============================================================================

def onesided_db(sig, fs=1.0, normalize=True):
    """Return (freqs, magnitude in dB) of the one-sided FFT."""
    sig = np.asarray(sig).ravel()
    N = len(sig)
    if normalize:
        sig = sig / np.max(np.abs(sig))
    fft_mag = np.abs(np.fft.rfft(sig)) / N
    fft_mag[1:-1] *= 2
    fft_db = 20 * np.log10(fft_mag + 1e-12)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    return freqs, fft_db


def plot_fft_onesided(sig, fs=1.0, title="FFT"):
    """Plot one-sided FFT magnitude spectrum in dB using Plotly."""
    freqs, fft_db = onesided_db(sig, fs)

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


def plot_fft_onesided_pair(sig1, sig2, fs=1.0, labels=("Signal 1", "Signal 2"),
                           title="FFT", normalize=True):
    """Plot the one-sided FFT magnitude of two signals in dB using Plotly."""
    colors = ("#1D9E75", "#E8833A")

    fig = go.Figure()
    for sig, label, color in zip((sig1, sig2), labels, colors):
        freqs, fft_db = onesided_db(sig, fs, normalize)
        fig.add_trace(go.Scatter(
            x=freqs, y=fft_db, mode="lines", name=label,
            line=dict(color=color, width=1.5)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    fig.show()
    return fig


def add_fft_trace(fig, sig, fs=1.0, title="FFT", name="", color="green", width=1.5):
    """Add the one-sided FFT magnitude spectrum of sig (in dB) to an existing figure."""
    freqs, fft_db = onesided_db(sig, fs)

    fig.add_trace(go.Scatter(
        x=freqs, y=fft_db, mode="lines",
        line=dict(color=color, width=width),
        name=name
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_dark",
        hovermode="x unified"
    )


def plot_phase_onesided(sig, fs=1.0, title="Phase Spectrum"):
    """Plot one-sided FFT phase spectrum in degrees using Plotly, with unwrapping."""
    N        = len(sig)
    sig_n    = sig / np.max(np.abs(sig))
    fft_vals = np.fft.rfft(sig_n)
    phase    = np.unwrap(np.angle(fft_vals))  # unwrap in radians first
    phase    = np.degrees(phase)              # then convert to degrees
    freqs    = np.fft.rfftfreq(N, d=1 / fs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=phase, mode="lines",
        line=dict(color="#E8833A", width=1.5)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Phase (degrees)",
        template="plotly_dark",
        hovermode="x unified"
    )
    fig.show()


def interpolate_x(x1, y1, x2, y2, y_target):
    """Linear interpolation to find x where y = y_target."""
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)


def plot_fft_onesided_n_modes(sig, fs=1.0, title="FFT", n_modes=3):
    """Plot one-sided FFT magnitude spectrum in dB.

    Returns interpolated resonance frequencies and -3 dB frequencies.
    """
    freqs, fft_db = onesided_db(sig, fs)

    # --------------------------------------------------
    # Find peaks
    # --------------------------------------------------

    peaks, _ = find_peaks(
        fft_db,
        prominence=3
    )

    # Sort peaks by magnitude
    peaks = peaks[np.argsort(fft_db[peaks])[::-1]]

    # Select strongest peaks
    peaks = peaks[:n_modes]

    # Sort by frequency
    peaks = np.sort(peaks)

    # Results
    results = []

    # --------------------------------------------------
    # Process each resonance
    # --------------------------------------------------

    for i, peak in enumerate(peaks):

        # ==================================================
        # Interpolated resonance frequency
        # ==================================================

        if 0 < peak < len(fft_db) - 1:

            y1 = fft_db[peak - 1]
            y2 = fft_db[peak]
            y3 = fft_db[peak + 1]

            delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)

            df = freqs[1] - freqs[0]

            f0 = freqs[peak] + delta * df

            peak_db = y2 - 0.25 * (y1 - y3) * delta

        else:
            f0 = freqs[peak]
            peak_db = fft_db[peak]

        # ==================================================
        # -3 dB level
        # ==================================================

        db_3 = peak_db - 3.0

        # ==================================================
        # Left -3 dB crossing
        # ==================================================

        left = peak

        while left > 0 and fft_db[left] > db_3:
            left -= 1

        if left > 0:
            f_low = interpolate_x(
                freqs[left],
                fft_db[left],
                freqs[left + 1],
                fft_db[left + 1],
                db_3
            )
        else:
            f_low = freqs[0]

        # ==================================================
        # Right -3 dB crossing
        # ==================================================

        right = peak

        while right < len(fft_db) - 1 and fft_db[right] > db_3:
            right += 1

        if right < len(fft_db) - 1:
            f_high = interpolate_x(
                freqs[right - 1],
                fft_db[right - 1],
                freqs[right],
                fft_db[right],
                db_3
            )
        else:
            f_high = freqs[-1]

        # ==================================================
        # Store results
        # ==================================================

        results.append({
            "frequency": f0,
            "f_3db_low": f_low,
            "f_3db_high": f_high,
            "bandwidth_3db": f_high - f_low,
            "peak_db": peak_db
        })

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=fft_db,
        mode="lines",
        name="FFT",
        line=dict(color="#1D9E75", width=1.5)
    ))

    for i, result in enumerate(results):

        f0 = result["frequency"]
        f_low = result["f_3db_low"]
        f_high = result["f_3db_high"]
        peak_db = result["peak_db"]
        db_3 = peak_db - 3

        # Resonance
        fig.add_trace(go.Scatter(
            x=[f0],
            y=[peak_db],
            mode="markers+text",
            name=f"Mode {i+1}",
            text=[f"f{i+1} = {f0:.2f} Hz"],
            textposition="top center",
            marker=dict(size=9)
        ))

        # -3 dB line
        fig.add_trace(go.Scatter(
            x=[f_low, f_high],
            y=[db_3, db_3],
            mode="lines+markers",
            name=f"Mode {i+1} -3 dB",
            line=dict(dash="dash")
        ))

        # -3 dB frequencies
        fig.add_vline(
            x=f_low,
            line_dash="dot",
            annotation_text=f"{f_low:.2f} Hz"
        )

        fig.add_vline(
            x=f_high,
            line_dash="dot",
            annotation_text=f"{f_high:.2f} Hz"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()

    return results


def plot_singular_values(S_a, S_b, labels=("Lagged", "Stacked"),
                         title="Singular values (lagged vs. stacked)",
                         log_y=False, normalize=True):
    colors = ("#1D9E75", "#E8833A")

    fig = go.Figure()

    if normalize:
        # ── Normalized — single shared y-axis ────────────────────
        for S, label, color in zip((S_a, S_b), labels, colors):
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(S) + 1),
                y=S / np.max(S),
                mode="lines", name=label,
                line=dict(color=color, width=1.5),
            ))
        fig.update_layout(
            yaxis=dict(title="Normalized magnitude")
        )

    else:
        # ── Not normalized — separate y-axes ─────────────────────
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(S_a) + 1),
            y=S_a,
            mode="lines", name=labels[0],
            line=dict(color=colors[0], width=1.5),
            yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(S_b) + 1),
            y=S_b,
            mode="lines", name=labels[1],
            line=dict(color=colors[1], width=1.5),
            yaxis="y2"
        ))
        fig.update_layout(
            yaxis=dict(
                title=dict(text=labels[0], font=dict(color=colors[0])),
                tickfont=dict(color=colors[0]),
                side="left",
            ),
            yaxis2=dict(
                title=dict(text=labels[1], font=dict(color=colors[1])),
                tickfont=dict(color=colors[1]),
                overlaying="y",
                side="right",
            ),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Singular value index",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    if log_y:
        fig.update_yaxes(type="log")

    fig.show()
    return fig


def plot_paired_branches_fft(outs, fs=1.0, normalize=False, group_size=2,
                             title="FFT of grouped branches (Hankel SVD)",
                             ylim=None, xlim=None):
    """
    Sum the branches of `outs` in groups of `group_size` and overlay their FFTs.

    Args:
        outs       : array (n_samples, n_branches), one column per branch
        group_size : number of consecutive branches summed together
                     1 -> one FFT per branch
                     2 -> branches 1+2, 3+4, ...
                     n_branches -> a single FFT of the total sum
        normalize  : passed to onesided_db
    """
    branches = outs.T                      # one row per branch
    n_branches = len(branches)

    if group_size < 1 or group_size > n_branches:
        raise ValueError(f"group_size must be between 1 and {n_branches}, got {group_size}")

    colors = ["#1D9E75", "#E8833A", "#5B8DEF", "#D6538C",
              "#F5C518", "#8E6CCF", "#4DB6AC", "#A1887F"]

    fig = go.Figure()
    n_groups = int(np.ceil(n_branches / group_size))   # last group may be smaller

    for g in range(n_groups):
        start = g * group_size
        end   = min(start + group_size, n_branches)    # exclusive
        sig   = branches[start:end].sum(axis=0)

        # Label: "Branch 3" for a single one, "Branches 1+2" or "Branches 1–4" for groups
        idx = list(range(start + 1, end + 1))
        if len(idx) == 1:
            label = f"Branch {idx[0]}"
        elif len(idx) <= 3:
            label = "Branches " + "+".join(map(str, idx))
        else:
            label = f"Branches {idx[0]}–{idx[-1]}"

        freqs, fft_db = onesided_db(sig, fs, normalize)
        fig.add_trace(go.Scatter(
            x=freqs, y=fft_db, mode="lines",
            name=label,
            line=dict(color=colors[g % len(colors)], width=1.5),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    if ylim is not None:
        fig.update_yaxes(range=list(ylim))
    if xlim is not None:
        fig.update_xaxes(range=list(xlim))
    fig.show()
    return fig


def plot_cumulative_branches_fft(outs, fs=1.0, normalize=False, step=1,
                                 title="FFT of cumulative branches (Hankel SVD)",
                                 ylim=None, xlim=None):
    """
    Overlay the FFT of the cumulative sum of branches:
    curve 1 = branch 1, curve 2 = branches 1+2, curve 3 = branches 1+2+3, ...

    Args:
        outs      : array (n_samples, n_branches), one column per branch
        normalize : passed to onesided_db
        step      : number of branches added between consecutive curves
                    1 -> one curve per branch (1, 1+2, 1+2+3, ...)
                    2 -> curves for 2, 4, 6, ... branches
    """
    branches   = outs.T                      # one row per branch
    n_branches = len(branches)

    if step < 1 or step > n_branches:
        raise ValueError(f"step must be between 1 and {n_branches}, got {step}")

    # Number of branches included in each curve (always ends at n_branches)
    counts = list(range(step, n_branches + 1, step))
    if counts[-1] != n_branches:
        counts.append(n_branches)

    # Sequential colorscale so the order of accumulation is visible
    palette = px.colors.sample_colorscale("Viridis",
                                          np.linspace(0, 0.9, len(counts)))

    fig = go.Figure()
    for c, color in zip(counts, palette):
        sig = branches[:c].sum(axis=0)

        label = "Branch 1" if c == 1 else f"Branches 1–{c}"

        freqs, fft_db = onesided_db(sig, fs, normalize)
        fig.add_trace(go.Scatter(
            x=freqs, y=fft_db, mode="lines",
            name=label,
            line=dict(color=color, width=1.5),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    if ylim is not None:
        fig.update_yaxes(range=list(ylim))
    if xlim is not None:
        fig.update_xaxes(range=list(xlim))
    fig.show()
    return fig


def compare_filters(svd_filt, fir_filt, n_samples, fs=1.0, ref=None,
                    title="SVD FIR vs. conventional FIR (impulse response)"):
    # Unit impulse
    x = np.zeros(n_samples)
    x[0] = 1.0

    y_svd = run_filter(svd_filt, x)
    y_fir = run_filter(fir_filt, x)
    err = y_svd - y_fir

    # Metrics
    mse = np.mean(err ** 2)
    max_err = np.max(np.abs(err))
    snr_db = 10 * np.log10(np.sum(y_fir ** 2) / (np.sum(err ** 2) + 1e-30))
    print("=" * 50)
    print(f"{'IMPULSE RESPONSE ERROR':^50}")
    print("=" * 50)
    print(f"{'MSE':<20}{mse:>20.3e}")
    print(f"{'Max |error|':<20}{max_err:>20.3e}")
    print(f"{'SNR (FIR vs. error)':<20}{snr_db:>17.2f} dB")
    if ref is not None:
        ref = np.asarray(ref).ravel()[:n_samples]
        print(f"{'Max |FIR - ref|':<20}{np.max(np.abs(y_fir - ref)):>20.3e}")
    print("=" * 50)

    # Plot
    t = np.arange(n_samples) / fs
    xlabel = "Time (s)" if fs != 1.0 else "Samples"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Impulse responses", "Error (SVD - FIR)"))
    fig.add_trace(go.Scatter(x=t, y=y_fir, mode="lines", name="Conventional FIR",
                             line=dict(color="#0F7B5F", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y_svd, mode="lines", name="SVD FIR",
                             line=dict(color="#D9650F", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=err, mode="lines", name="Error",
                             line=dict(color="#2A5BD7", width=1.5)), row=2, col=1)

    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        height=650,
    )
    fig.show()
    return fig, err


# =============================================================================
# RESOURCE UTILIZATION
# =============================================================================

def calc_svd_utilizations(C_weights, R_weights):
    """SVD FIR memory, coefficients and operations from its weight matrices."""
    R = R_weights.shape[1]
    C = C_weights.shape[1]
    N_branches = C_weights.shape[0]

    num_coefs = N_branches* (R + C)
    num_ops = (2*(R + C)*N_branches )+ N_branches - 1
    n_mem = R*C + num_coefs + C + N_branches*R
    print(f'{N_branches = } , {R = },{C = }, {num_coefs = }')
    return n_mem, num_coefs ,num_ops


def calc_svd_utilizations_teo(W_size, N_branches):
    """
    Return memory, coefficients and number of operations necessary for
    the svd operation by receiving as input the filter size and branch.
    Will assume the best R and C relation is the square of the filter_size.
    """
    R = int(np.ceil(np.sqrt(W_size))) # Size related to US
    C = int(np.ceil(W_size/R)) # Assumes padding (size related to Vt)

    num_coefs = N_branches* (R + C)
    num_ops = (2*(R + C)*N_branches )+ N_branches - 1
    n_mem = R*C + num_coefs + C + N_branches*R
    print(f'{W_size =}, {N_branches = } , {R = },{C = }, {num_coefs = }')
    return n_mem, num_coefs ,num_ops


def calc_fir_utilization(W_size):
    """Calculate common FIR resources utilizations with the input being the filter_size."""
    num_coefs = W_size
    n_mem = 2 * W_size
    n_ops = 2*W_size - 1
    return n_mem,num_coefs,n_ops

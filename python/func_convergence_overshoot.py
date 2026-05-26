#%%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from CantileverBeam import CantileverBeam
from Adaptive import FIRNLMS
from AdaptiveOO import FIRFxNLMS, FIR
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
#%%

fs = 416.0 # Sampling frequency in Hertz

# Beam characteristics:
npoints = 100 # Number of points in the beam (finite element method)
beamlength = 0.58 # Length of the beam in meters
beamwidth = 0.05 # Width of the beam in meters
beamthickness = 0.006 # Thickness of the beam in meters
dampingfactors = [0.01, 0.01, 0.01, 0.01, 0.01] 


# Positions of sensors and forces:
perturbpos = 30 # Position of the perturbation force, which causes beam vibration.
referencepos = 75 # Position of the acceleration measurement at the beam.
controlpos = 60 # Position of the control force
errorpos = 95 # Position of the error acceleration measurement in the beam

firmem = 500 # Number of samples for the secondary and feedback paths


#  Creating Beam instance with 100 points:
cbeam = CantileverBeam(npoints=npoints, width=beamwidth, thickness=beamthickness, 
                        length=beamlength, Tsampling=1.0/fs,
                        damp=dampingfactors)
cbeam.reset()
print("Natural frequencies are:\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()),
      " (all in Hz).")

xcoords = np.linspace(0.0, cbeam.length, npoints)
xcoords = np.concatenate((xcoords, xcoords[::-1]))
ycoords = np.array([0.0]*npoints + [beamthickness]*npoints)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xcoords, y=ycoords, fill='toself', mode='lines'))
fig.add_annotation(x=perturbpos*beamlength/npoints, y=beamthickness*1.1, 
            ax=0, ay=-30, text="Perturbation",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=referencepos*beamlength/npoints, y=beamthickness*1.1, 
            ax=0, ay=-50, text="Accel. Measurement",
            showarrow=True, arrowhead=1, arrowside="start")
fig.add_annotation(x=controlpos*beamlength/npoints, y=0, 
            ax=0, ay=30, text="Control Force",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=errorpos*beamlength/npoints, y=0, 
            ax=0, ay=50, text="Error Accel.",
            showarrow=True, arrowhead=1, arrowside="start")
fig.update_layout(title="Cantilever Beam", xaxis_title="x (m)", yaxis_title="y (m)")
fig.update_layout(xaxis=dict(range=[0, beamlength*1.1]), yaxis=dict(range=[-(beamthickness + 0.1), beamthickness + 0.1]))
fig.update_layout(width=600, height=350)
fig.show()




maxtime = 60.0
vibstart = 10.0 # Start time of the vibration
nsteps = int(maxtime * fs) # Total number of steps
vibfreq = 10.0 # Hertz
th = np.linspace(0.0,maxtime,nsteps) # Time vector
xh = 0.3*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force set to zero for the first 10 seconds

cbeam.reset()
err = np.zeros(nsteps) # Vibration response
# Running simulation:
for k in range(nsteps):
  err[k] = cbeam.getaccelms2(referencepos)  
  cbeam.setforce(perturbpos,xh[k]) 
  cbeam.update() # Updata for 1 sampling period.

# Plotting the results:
fig = px.line()
fig.add_scatter(x=th, y=xh, name="Força (N)", mode="lines")
fig.add_scatter(x=th, y=err, name="Aceleração (m/s²)", mode="lines")
fig.show()



# Matrix formatting with zero padding:
def format_W(W,R):
    n_pad = W.shape[0] % R
    W_pad = np.zeros(int(W.shape[0] + (R - n_pad)))
    W_pad[:W.shape[0]] = W
    C = W_pad.shape[0]/R
    lower_dim = min(R,C)
    return  W_pad.reshape((int(lower_dim),-1),order = 'F')

# FIRSVD Implementation: 
class FIRSVDFilterPy(FIR):
    def __init__(self, C_weights, R_weights):
        self.R  =  R_weights.shape[1] # Rows
        self.C  =  C_weights.shape[1] # Collunms
        self.B  =  C_weights.shape[0] # Number of bases
        self.N = self.R*self.C
        self.vdot = C_weights        
        self.inputbuffer = np.zeros(self.N)
        self.util = []
        for k in range(self.B):
           self.util.append(FIR(R_weights[k,:]))
        self.reset()

    def reset(self):
        self.y = 0
        self.inputbuffer[:] = 0.0
        for k in range(self.B):
            self.util[k].reset()
            
    def filterstep(self, xn):
        self.inputbuffer[1:] = self.inputbuffer[:-1]
        self.inputbuffer[0] = xn
        self.youts = self.vdot @ self.inputbuffer[::self.R]
        for k in range(self.B):
            self.youts[k] = self.util[k].filterstep(self.youts[k])
        self.y = np.sum(self.youts)
        return self.y


def gen_wsec_wfbk_filters(firmem):
  # Secondary path via impulse response (ideal but not practical):
  wsecimpulse = np.zeros(firmem) # Impulse response vector
  cbeam.reset()
  cbeam.setforce(controlpos,1.0) # Force is applied at the control position
  cbeam.update()
  wsecimpulse[0] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position
  cbeam.setforce(controlpos,0.0) # Force is removed
  for k in range(1,firmem):
    cbeam.update() # Update the beam for 1 sampling period.
    wsecimpulse[k] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position


  wfbkimpulse = np.zeros(firmem) # Impulse response vector
  cbeam.reset()
  cbeam.setforce(controlpos,1.0) # Force is applied at the control position
  cbeam.update()
  wfbkimpulse[0] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position
  cbeam.setforce(controlpos,0.0) # Force is removed
  for k in range(1,firmem):
    cbeam.update() # Update the beam for 1 sampling period.
    wfbkimpulse[k] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position

  fig = px.line()
  fig.add_scatter(y = wfbkimpulse)
  fig.add_scatter(y = wsecimpulse)
  
  return wsecimpulse,wfbkimpulse

#Implement SVD with 300 equivalent number of coefficients and compare with df_wsec_wfbk 

"""
Calculate and organize filter weights for SVD
"""
def gen_SVD_weights(weights,num_coef):
    """
    Args:
        weights (array): Array of filters weights
        num_coef: Number of coefficiets desired for SVD filter num_branches*( num_collunms + num_rows)
    """
    
    #Will assume the most squared matrix for the weights
    full_size = len(weights)
    C_chosen_s = int(np.sqrt(full_size))

    W = format_W(weights,C_chosen_s)
    print(f'{W.shape = }')
    R_s = W.shape[0]
    B_s = int(np.round(num_coef/(C_chosen_s + R_s)))

    U,S,VT = np.linalg.svd(W)

    SM = np.zeros((R_s,C_chosen_s))
    np.fill_diagonal(SM,S)
    US = U @ SM
    C_weights = np.zeros((B_s,VT.shape[1]))
    R_weights = np.zeros((B_s,U.shape[0]))

    for i in range(B_s):
        C_weights[i,:] = VT.T[:,i]
        R_weights[i,:] = US[:,i]

    return C_weights,R_weights


# %%

"""
SVD FILTER FPGA comparison:

- Generate weights here: 
- Create file with Vt and US parameters for all branchess
- run pythoon simulation with random samples, get input sequence and expected output in a file
- testbench with all zeros first, then sequence.
- Save output in textfile (Or make the quantization in systemverilog, not good for plotting)
"""

firmem = 6400
mu = 0.01
wsecimpulse , wfbkimpulse = gen_wsec_wfbk_filters(firmem)
print(wsecimpulse.shape,wfbkimpulse.shape)
filter_size = 256
C_weights_sec,R_weights_sec = gen_SVD_weights(wsecimpulse,filter_size)
print(C_weights_sec.shape,R_weights_sec.shape)

svd_filter_py = FIRSVDFilterPy(C_weights_sec,R_weights_sec)
svd_filter_py.reset()

x = np.zeros(1000) 
x[0] = 1
y = np.zeros_like(x)
for i,sample in enumerate(x): 
    y[i] = svd_filter_py.filterstep(sample)

def plot_fft_onesided(signal, fs=1.0, title="FFT"):
    N       = len(signal)
    signal_n = signal/np.max(np.abs(signal))
    fft_mag = np.abs(np.fft.rfft(signal_n)) / N      # rfft gives positive freqs only
    fft_mag[1:-1] *= 2                              # double to conserve power
    fft_db  = 20 * np.log10(fft_mag + 1e-12)
    freqs   = np.fft.rfftfreq(N, d=1/fs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=fft_db,
        mode="lines",
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
plot_fft_onesided(y,fs=400)
# %%
fig = px.line()
fig.add_scatter(y = x, name="x")
fig.add_scatter(y = y, name="y")
fig.show()
# Fixed-point quantization — scale to fit WIDTH=8 bits signed (-128..127)
SCALE = 127.0 / np.max(np.abs(x))
quantized = np.clip(np.round(x * SCALE), -128, 127).astype(np.int8)

with open("samples.hex", "w") as f:
    for s in quantized:
        # & 0xFF interprets signed byte as unsigned for hex representation
        f.write(f"{int(s) & 0xFF:02X}\n")

# %%
def quantize_matrix(matrix,max_ov, width=8):
    """Quantize a float matrix to signed fixed-point integers."""
    #max_val = np.max(np.abs(matrix))
    scale = (2**(width-1) - 1) / max_ov if max_ov != 0 else 1
    quantized = np.clip(np.round(matrix * scale), -(2**(width-1)), 2**(width-1)-1)
    return quantized.astype(np.int8)

def matrix_to_sv_hex(matrix, name, width=16):
    if (len(matrix.shape) == 2):
        rows, cols = matrix.shape
    else :
        cols , rows = len(matrix) , 1
    hex_digits = width // 4  # nibbles per element

    print(f"// {name} — shape ({rows}, {cols})")
    for r in range(rows):
        # concatenate all column hex values in one string
        if rows != 1 :
            row_hex = "".join(f"{int(v) & (2**width-1):0{hex_digits}X}" for v in reversed(matrix[r]))
        else:
            row_hex = "".join(f"{int(v) & (2**width-1):0{hex_digits}X}" for v in reversed(matrix))
            
        total_bits = cols * width
        print(f"// ({r}, :)")
        print(f"parameter [{total_bits-1}:0] {name}_ROW{r} = {total_bits}'h{row_hex};")
    print()

# ── Your matrices ──────────────────────────────────────────────────
max_ov = np.max([np.max(np.abs(C_weights_sec)),np.max(np.abs(R_weights_sec))])
Vt_q = quantize_matrix(C_weights_sec,max_ov)
Us_q = quantize_matrix(R_weights_sec,max_ov)
fig = px.line()
fig.add_scatter(y = Vt_q[0,:], name="Vt b=0")
fig.add_scatter(y = Us_q[0,:], name="Us b=0")
fig.show()
matrix_to_sv_hex(Vt_q, "VT")
matrix_to_sv_hex(Us_q, "US")
# %%
#Load results.txt file and plot
df = pd.read_csv("../work/results.txt", skipinitialspace=True)
fig = px.line()
#fig.add_scatter(y = df['x'], name="x")
fig.add_scatter(y = df['y'], name="y")
fig.show()

plot_fft_onesided(df['y'],fs=400)
plot_fft_onesided(y,fs=400)


# %%
def norm(x):
   return x/np.max(np.abs(x))
fig = px.line()
fig.add_scatter(y = norm(FIR_w), name="y python")
fig.add_scatter(y = norm(df['y']), name="y FPGA")
#fig.add_scatter(y = norm(y) - norm(df['y']) , name="Error")

fig.show()
# %%
"""
Create original FIR filter in FPGA to compare with the svd results
"""
fir_weights = wsecimpulse
print(fir_weights.shape)
max_ov = np.max(np.abs(fir_weights))
FIR_w = quantize_matrix(fir_weights,max_ov)
print(FIR_w.shape)

# %%
fig = px.line()
fig.add_scatter(y = FIR_w, name="FIR weights quantized")
matrix_to_sv_hex(FIR_w, "W")
print(FIR_w)
# %%

def weights_to_hex(weights: np.ndarray, filename: str, width: int = 8) -> None:
    """
    Quantize a numpy array of filter weights and write to a .hex file
    for use with $readmemh in SystemVerilog.

    Args:
        weights  : numpy array of any shape (will be flattened row-major)
        filename : output .hex file path
        width    : bit width of each coefficient (default 8)
    """
    max_val = np.max(np.abs(weights))
    scale   = (2**(width-1) - 1) / max_val if max_val != 0 else 1.0

    quantized = np.clip(
        np.round(weights.flatten() * scale),
        -(2**(width-1)),
         (2**(width-1)) - 1
    ).astype(np.int8 if width <= 8 else np.int16)

    hex_digits = width // 4  # nibbles per value

    with open(filename, "w") as f:
        for v in quantized:
            # & mask interprets signed as unsigned for hex printing
            f.write(f"{int(v) & (2**width - 1):0{hex_digits}X}\n")

    print(f"Written {filename} — {len(quantized)} entries "
          f"(scale={scale:.4f}, range=[{quantized.min()}, {quantized.max()}])")


# %%

weights_to_hex(FIR_w , "../weights/FIR_weights.hex" , width=16)
# %%
import os
print(os.getcwd())
# %%

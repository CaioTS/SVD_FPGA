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

def calculate_convergence_time (err, threshold):
    
    err = np.clip(err,-1e6, 1e6)
    energy  = err**2
    fs = 416
    th = np.linspace(0.0,len(err)/fs,len(err)) # Time vector

    max_energy = max(energy[int(fs*10): int(fs*20)])
    max_index_original = np.argmax(energy[int(fs*10): int(fs*20)]) + int(fs*10)

    overshoot = max(energy[int(fs*30):])/max_energy
    overshoot_index = np.argmax(energy[int(fs*30):]) + int(fs*30)
    indeces = np.arange(0,len(th),1)

    indeces_above = indeces[energy[indeces] >= threshold * max_energy]
    point_above = indeces_above[-1] + 1 
    if point_above >= len(th):
       point_above = len(th) - 1
    #Get times to Reach the threshold of energy

    dt = th[indeces_above[-1]] - 30
    return overshoot,dt


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


def run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD=False,compressed_filters='both'):
    """
    Simulation for the beam and different filters implementations

    Args:
        pertb_freq (float): Frequency of the perturbation signal in Hz.
        pertb_amp (float): Amplitude of the perturbation signal.
        threshold (float): Convergence threshold for the error signal. Defaults to 0.05.
        filter_size (int): Number of taps (memory length) for the FIR filter.
        wsecimpulse (array_like): Impulse response of the secondary path (S).
        wfbkimpulse (array_like): Impulse response of the feedback path (F).
        mu (float): Step size (learning rate) for the adaptive algorithm update.
        psi (float): Regularization parameter.
        isSVD (bool): If True, performs the update using Singular Value Decomposition.
        compressed_filters(string): Choose what filters are compressed: (both, sec, fbk)
    Returns:
        tuple: A tuple containing (overshoot , convergence_time).
    """
    np.seterr(over='raise', invalid='raise', divide='raise') #Operation Warning now are caught by try except

    cbeam.reset()
    force_amplitude = pertb_amp
    maxtime = 120.0
    nsteps = int(maxtime * fs) # Total number of steps
    vibstart = 0.0 # Start time of the vibration
    controlstart = 30.0 # Start time of the control
    
    if compressed_filters == "both":
        wsec = wsecimpulse[:filter_size]
        wfbk = wfbkimpulse[:filter_size]
        
    elif compressed_filters == "sec":
        wsec = wsecimpulse[:filter_size]
        wfbk = wfbkimpulse
    elif compressed_filters == "fbk":
        wsec = wsecimpulse
        wfbk = wfbkimpulse[:filter_size]
    else:
        raise ValueError("compressed filter not compatible with function")


    if isSVD :
       C_weights_sec,R_weights_sec = gen_SVD_weights(wsecimpulse,filter_size)
       C_weights_fbk,R_weights_fbk = gen_SVD_weights(wfbkimpulse,filter_size)

    controller = FIRFxNLMS(mem=filter_size, memsec=filter_size) # Create the controller
    # controller.setSecondary(wsecimpulse) # Set the secondary path
    if isSVD :
      controller.setSecondary(FIRSVDFilterPy(C_weights_sec,R_weights_sec)) # Set the secondary path
    else:
      controller.setSecondary(FIR(wsec)) # Set the secondary path
    controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
    controller.mu = mu # Set the step size
    controller.psi = psi # Set the regularization parameter
    controller.reset() # Reset the controller
    if isSVD :
      feedbackfilter = FIRSVDFilterPy(C_weights_fbk,R_weights_fbk)  # Create the feedback filter
    else:
      feedbackfilter = FIR(wfbk)
    feedbackfilter.reset() # Reset the filter

    vibfreq = pertb_freq # Hertz
    th = np.linspace(0.0,maxtime,nsteps) # Time vector
    xh = force_amplitude*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
    xh[0:int(fs*vibstart)] = 0.0 # Force is zero for the first 10 seconds

    
    err = np.zeros(nsteps) # Vibration response
    yfbk = np.zeros(nsteps) # Vibration response
    sim_status = True
    # Running the simulation:
    for k in range(nsteps):
        try:
            cbeam.setforce(perturbpos,xh[k]) # force is applied
            cbeam.setforce(controlpos,-controller.y) # Control force is applied

            if th[k] >= controlstart: # Control starts at 30 seconds
                controller.update(cbeam.getaccelms2(errorpos)) 
            yfbk[k] = feedbackfilter.filterstep(-controller.y) # Get the feedback force

            controller.evalout(cbeam.getaccelms2(referencepos) - yfbk[k])
        
        except (ValueError, FloatingPointError, OverflowError) as ex:
           print(f"Warning: numerical error at step {k}: {ex}")
           #Added symbolic values informing it failed
           ov = 1000 
           dt = 90
           sim_status = False
           break
            

        err[k] = cbeam.getaccelms2(errorpos) # Error acceleration is read

        cbeam.update() # beam is updated
    if sim_status :
        ov, dt = calculate_convergence_time(err,threshold)
        if ov > 1000:
           ov = 1000
    fig = px.line(title=f"Vibration Control nmem = {filter_size} mu = {mu}",x=th,y=err)
    fig.add_vline(x=dt + 30, line_width=1, line_dash="dash", line_color="yellow")
    fig.add_hline(y=np.sqrt(ov), line_width=1, line_dash="dash", line_color="red")
    fig.show()
    return ov,dt

#%%

#Steps to use this function , Generate Beam(First lines in file)
#Generate full secondary and feedback weights
firmem = 6400
mu = 0.01
wsecimpulse , wfbkimpulse = gen_wsec_wfbk_filters(firmem)
pertb_freq , pertb_amp = 13.5 , 0.3
threshold = 0.05
filter_size = 200
mu = 0.008
psi = 1e-2
ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD =False,compressed_filters='both')
ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD =True,compressed_filters='sec')
ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD =False,compressed_filters='sec')
# %%
"""
Debug overshoot issue with specfic parameters sent by Dudu
filter_size = 200
pertb_freq = 14
pertb_amp = 0.3
mu = range
"""
filter_size = 200
pertb_freq = 14
pertb_amp = 0.3
mu = 0.04

ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD =False,compressed_filters='both')
ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,mu,psi,isSVD =True,compressed_filters='both')

# %%
mus = [0.001 , 0.002,0.005, 0.006, 0.008, 0.01,0.02,0.03,0.04,0.06,0.08]
ov_svd, dt_svd = [] , []
ov_nom, dt_nom = [] , []
for m in mus:
   ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,m,psi,isSVD =True,compressed_filters='both')
   ov_svd.append(ov)
   dt_svd.append(dt)
   ov , dt = run_sim(pertb_freq,pertb_amp,threshold,filter_size,wsecimpulse,wfbkimpulse,m,psi,isSVD =False,compressed_filters='both')
   ov_nom.append(ov)
   dt_nom.append(dt)



# %%
fig = px.line()
fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Time of Convergence (s)"
)
fig.add_scatter(y=dt_svd, x = mus, name='SVD')
fig.add_scatter(y=dt_nom, x = mus, name= 'non-SVD')

fig.show()

fig = px.line()
fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Overshoot"
)
fig.add_scatter(y=ov_svd, x = mus, name='SVD')
fig.add_scatter(y=ov_nom, x = mus, name= 'non-SVD')

fig.show()
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
    rows, cols = matrix.shape
    hex_digits = width // 4  # nibbles per element

    print(f"// {name} — shape ({rows}, {cols})")
    for r in range(rows):
        # concatenate all column hex values in one string
        row_hex = "".join(f"{int(v) & (2**width-1):0{hex_digits}X}" for v in reversed(matrix[r]))
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
fig.add_scatter(y = df['x'], name="x")
fig.add_scatter(y = df['y'], name="y")
fig.show()

plot_fft_onesided(df['y'],fs=400)
plot_fft_onesided(y,fs=400)


# %%
def norm(x):
   return x/np.max(np.abs(x))
fig = px.line()
fig.add_scatter(y = norm(y), name="y python")
fig.add_scatter(y = norm(df['y']), name="y FPGA")
fig.add_scatter(y = norm(y) - norm(df['y']) , name="Error")

fig.show()
# %%

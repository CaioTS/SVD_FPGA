# %%
# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal

from AdaptiveOO import FIR
from aux_functions import gen_SVD_weights, FIRSVDFilterPy, add_fft_trace, lagged_matrix


#%%
fs = 416  # sampling frequency
n0, f0_c = 40, 40  # number of taps and cut-off frequency for H_0(f)

b0 = signal.firwin(n0, f0_c, fs=fs)  # design FIR filters

f0, H0 = signal.freqz(b0, fs=fs)  # calculate frequency responses
H0_dB = 20 * np.log10(np.abs(H0))

fig = go.Figure()

# Frequency response
fig.add_trace(
    go.Scatter(
        x=f0,
        y=H0_dB,
        mode="lines",
        name=r"$|H_0(f)|$"
    )
)
# Cutoff frequency
fig.add_vline(
    x=f0_c,
    line_dash="dash",
    line_color="green",
    annotation_text="Cutoff Frequency",
    annotation_position="top right"
)
fig.update_layout(
    title=f"Frequency Response of {n0}-tap Low-pass FIR Filter",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Gain (dB)",
    xaxis=dict(range=[0, fs/2]),
    template="plotly_white",
    legend=dict(x=0.01, y=0.99)
)
fig.show()

fig = px.line(y=b0)
fig.show()


# %%
#Create SVD step and generate filter
svd_size = 6* (7 + 6)
C_weights, R_weights , S= gen_SVD_weights(b0,svd_size , C_size=7, verbose=False)
print(C_weights.shape,svd_size)
svd_filter_py = FIRSVDFilterPy(C_weights, R_weights)
filter_py = FIR(b0)
svd_filter_py.reset()
filter_py.reset()

x = np.zeros(1000)
x[0] = 1.0
y_orig = np.zeros_like(x)
y = np.zeros_like(x)
for i, sample in enumerate(x):
    y[i] = svd_filter_py.filterstep(sample)
    y_orig[i] = filter_py.filterstep(sample)

fig = px.line()
fig.add_scatter(y=x, name="Input (impulse)")
fig.add_scatter(y=y, name="Output y")
fig.update_layout(title="Python SVD Filter — Impulse Response")
fig.show()

fig = go.Figure()
color = ('green', 'blue')
add_fft_trace(fig,y, fs=fs, title="FFT —  SVD and FIR Filter Output",name="SVD",color=color[0])
add_fft_trace(fig,y_orig, fs=fs, title="FFT — SVD and FIR Filter Output",name="FIR",color=color[1])
fig.show()

print(S.shape)
px.line(y=S).show()

#%%
"""
Execute and compare the FFT of the SVDs by removing one branch at a time
"""
x = np.zeros(1000)
x[0] = 1.0
y = np.zeros_like(x)
color = ('green','blue','red','yellow','orange','purple')
fig = go.Figure()
fig_single = go.Figure()
add_fft_trace(fig,y_orig, fs=fs, title="FFT —  SVD and FIR Filter Output",name="FIR",color="white",width=3)

for i in range(6):
    B = i+1
    svd_size = B* (7 + 6)
    C_weights, R_weights , S= gen_SVD_weights(b0,svd_size , C_size=7, verbose=False)
    svd_filter_py = FIRSVDFilterPy(C_weights, R_weights)
    svd_filter_py.reset()
    for j, sample in enumerate(x):
        y[j] = svd_filter_py.filterstep(sample)

    add_fft_trace(fig,y, fs=fs, title="FFT —  SVD and FIR Filter Output",name=f"SVD Branches = {B}",color=color[i])
    add_fft_trace(fig_single,np.array(svd_filter_py.youts_outs)[:,i], fs=fs, title="FFT —  Individual branches",name=f"SVD Branches = {B}",color=color[i])
fig.show()
fig_single.show()
# %%
#HANKELIZAtion
#Something interesting happened here: the eigen-values have a similar correspondence with the filters response filter. 


def Hankelization_analysis(weights,fs):

    f0, H0 = signal.freqz(weights, fs=fs)  # calculate frequency responses
    H0_dB = 20 * np.log10(np.abs(H0))

    # Create lagged matrice
    L = int(len(weights)/2)
    w_lagged = lagged_matrix(weights, L)
    #Make SVD of lagged vector
    U, S_lagged, VT = np.linalg.svd(w_lagged)
    print(U.shape , S_lagged.shape , VT.shape)
    fig = go.Figure()
    x = np.arange(0,fs/2, fs/(2*len(S_lagged)))
    # Left y-axis (log)
    fig.add_trace(go.Scatter(
        #x=x,
        y=S_lagged / np.max(S_lagged),
        mode="lines",
        name="Stacked"
    ))

    # Right y-axis
    #fig.add_trace(go.Scatter(
    #    x=f0,
    #    y=H0_dB,
    #    mode="lines",
    #    name="|H0(f)|",
    #    yaxis="y2"
    #))

    fig.update_layout(
        title="Normalized Eigenvalues of SVD on Lagged Matrices",
        xaxis_title="Index",
        yaxis=dict(
            title="Normalized Singular Value",
            type="log"
        ),
        yaxis2=dict(
            title="|H0(f)| (dB)",
            overlaying="y",
            side="right"
        ),
        template="plotly_white"
    )

    fig.show()
# %%
fs = 416  # sampling frequency
n0, f0_c = 400, 40  # number of taps and cut-off frequency for H_0(f)
b0 = signal.firwin(n0, f0_c, fs=fs)  # design FIR filters
Hankelization_analysis(b0,fs)
# %%
n0, f0_c = 800, 40  # number of taps and cut-off frequency for H_0(f)
b0 = signal.firwin(n0, f0_c, fs=fs)  # design FIR filters
Hankelization_analysis(b0,fs)
# %%

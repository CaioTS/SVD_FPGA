# %%
# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import plotly.graph_objects as go

from CantileverBeam import CantileverBeam
from aux_functions import (
    lagged_matrix, get_impulse_response, gen_wsec_wfbk_filters,
    onesided_db, plot_fft_onesided_pair,
)

# %%
# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
fs          = 600
npoints     = 100
beamlength  = 0.58
beamwidth   = 0.05
beamthickness = 0.006
dampingfactors = [0.01, 0.01, 0.01, 0.01, 0.01]

perturbpos  = 30
controlpos  = 60
errorpos    = 95
firmem      = 2400
mu          = 0.01

# Reference positions to compare
referencepos_list = [70, 72, 75, 77, 80]

# %%
# =============================================================================
# BEAM INSTANTIATION
# =============================================================================
cbeam = CantileverBeam(
    npoints=npoints,
    width=beamwidth,
    thickness=beamthickness,
    length=beamlength,
    Tsampling=1.0 / fs,
    damp=dampingfactors,
    noisestd=0.0,
    nmodes= 3
)
cbeam.reset()

print("Natural frequencies (Hz):\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()))

# %%
# =============================================================================
# FFT OF FEEDBACK PATH FOR DIFFERENT REFERENCE POSITIONS
# =============================================================================
# Computes wfbkimpulse for each referencepos and overlays their FFTs.

colors_ref = [
    "#1D9E75", "#E8833A", "#5B8DEF", "#E84393", "#F5C518"
]

fig_fft = go.Figure()

for idx, refpos in enumerate(referencepos_list):

    # Recompute feedback impulse for this reference position
    wfbk = get_impulse_response(cbeam, firmem, controlpos, refpos)

    # Compute one-sided FFT
    freqs, fft_db = onesided_db(wfbk, fs)

    fig_fft.add_trace(go.Scatter(
        x=freqs, y=fft_db,
        mode="lines",
        name=f"refpos = {beamlength * refpos:.2f} cm",
        line=dict(color=colors_ref[idx % len(colors_ref)], width=1.5)
    ))

fig_fft.update_layout(
    #title="Feedback Path FFT — different reference positions",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude (dB)",
    template="plotly_white",
    hovermode="x unified"
)

fig_fft.update_yaxes(range=list([-90 , -35]))
fig_fft.update_xaxes(range=list([60 , 120]))

fig_fft.show()
fig_fft.write_image("figs/feedback_ref_pos_change_fft.pdf") 

# %%
# =============================================================================
# FINAL IMPULSE RESPONSE WITH Controlled window
# =============================================================================
# Recompute with the original referencepos for use in the rest of the script.
referencepos = 75
wsecimpulse, wfbkimpulse = gen_wsec_wfbk_filters(cbeam, firmem, controlpos, errorpos, referencepos)

#Executes sweeps of head_len and executes the svd of the stacked matrix

fig = go.Figure()
for head_len in range(0,10,1):
    ntail = firmem - 100 - head_len
    L = int(np.ceil(np.sqrt(ntail)))
    K = int(np.ceil(ntail / L))
    nfir = head_len + L * K
    print(nfir , head_len , K , L)
    h = wfbkimpulse[head_len:nfir]
    h = h.reshape(K, L).T
    U , S , Vt  = np.linalg.svd(h)
    fig.add_trace(go.Scatter(y=S, mode="lines+markers", name=f"head_len = {head_len}"))

fig.update_layout(yaxis_type="log", showlegend=True)


fig.show()

# %%

#Executes sweeps of head_len and executes the svd of the hankel matrix
fig = go.Figure()
for head_len in range(0,10,1):
    ntail = firmem - 100 - head_len
    L = int(np.ceil(np.sqrt(ntail)))
    K = int(np.ceil(ntail / L))
    nfir = head_len + L * K
    print(nfir , head_len , K , L)
    h = wfbkimpulse[head_len:nfir]
    h_lag = lagged_matrix(h, L)[:, :K]   # (L, K) Hankel matrix
    #h = h.reshape(K, L).T
    U , S , Vt  = np.linalg.svd(h_lag)
    fig.add_trace(go.Scatter(y=S, mode="lines+markers", name=f"N = {head_len}"))

fig.update_layout(yaxis_type="log", showlegend=True)

fig.update_layout(
    xaxis_title="Singular Values",
    xaxis=dict(range=[0 , 10]),
    yaxis=dict(range=[-3 , 1]),
    template="plotly_white",
    hovermode="x unified"
    #legend=dict(groupclick="toggleitem")
)
fig.show()
fig.write_image("figs/S_values_head_len_fbk.pdf")

plot_fft_onesided_pair(wfbkimpulse[:nfir], wfbkimpulse[2:nfir], fs=fs, labels=("N = 0", "N = 2"), normalize=True)

# Run Control with only 6 branches com nulo perto mas não quebrando a ressonancia
#%%

# Avaliar mesmo experimento com situação crítica
# %%
#Fazer amostragem com 832 Hz 3 modos

#Fazer amostragem com 416 Hz 2 modos
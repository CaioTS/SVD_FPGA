#%%
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

"""
Return coefficients , memory and number of operations necessary for 
the svd operation by receveing as input the filter size and branch. 
Will assume the best R and C relation is the square of the filter_size
"""
def calc_svd_utilizations(W_size,N_branches):
    
    R = int(np.ceil(np.sqrt(W_size))) # Size related to US
    C = int(np.ceil(W_size/R)) # Assumes padding (size related to Vt)


    num_coefs = N_branches* (R + C)
    num_ops = (2*(R + C)*N_branches )+ N_branches - 1 
    n_mem = R*C + num_coefs + C + N_branches*R 
    print(f'{W_size =}, {N_branches = } , {R = },{C = }, {num_coefs = }')
    return n_mem, num_coefs ,num_ops

"""Calculate commmon FIR resources utilizations with the 
input being the filter_size"""
def calc_fir_utilization(W_size):
    num_coefs = W_size
    n_mem = 2 * W_size
    n_ops = 2*W_size - 1 
    return n_mem,num_coefs,n_ops

calc_svd_utilizations(400,2)



# %%
fs = 416

N = np.arange(500,2000,300)
B = np.arange(1,6,1)
svd_coefs = np.zeros_like(N)
svd_mems  = np.zeros_like(N)
svd_ops   = np.zeros_like(N)

fir_coefs = np.zeros_like(N)
fir_mems  = np.zeros_like(N)
fir_ops   = np.zeros_like(N)

for n in N:  
    svd_mems[N == n],svd_coefs[N == n], svd_ops[N == n] = calc_svd_utilizations(n,2)
    fir_mems[N == n],fir_coefs[N == n], fir_ops[N == n] = calc_fir_utilization(n)
# %%
compression_coefs = 100 * (svd_coefs/fir_coefs)
compression_mems  = 100 * (svd_mems /fir_mems)
compression_ops   = 100 * (svd_ops  /fir_ops)

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

colors = sample_colorscale(
    "Viridis",
    np.linspace(0, 1, len(B))
)

# Create subplots with 1 row and 3 columns
fig = make_subplots(
    rows=1, 
    cols=3,
    subplot_titles=(
        'Coefficients (coefs)', 
        'Memory (mems)', 
        'Operations (ops)'
    ),
    horizontal_spacing=0.1
)

# Add trace related to number of coefficients per branch and FIR
for c,b in enumerate(B):
    for n in N:
        svd_mems[N == n],svd_coefs[N == n], svd_ops[N == n] = calc_svd_utilizations(n,b)
    fig.add_trace(
        go.Scatter(x=N, y=svd_coefs, mode='lines+markers', name=f'Branches = {b}',  line=dict(color=colors[c])),
        row=1, col=1
    )

    fig.add_trace(
    go.Scatter(x=N, y=svd_mems, mode='lines+markers', name= f'Branches = {b}', line=dict(color=colors[c])),
    row=1, col=2
    )

    fig.add_trace(
    go.Scatter(x=N, y=svd_ops, mode='lines+markers', name= f'Branches = {b}', line=dict(color=colors[c])),
    row=1, col=3
)


#FIR in black for comparison
fig.add_trace(
    go.Scatter(x=N, y=fir_coefs, mode='lines+markers', name='fir_coefs', line=dict(color='black')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=N, y=fir_mems, mode='lines+markers', name='mems', line=dict(color='black')),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=N, y=fir_ops, mode='lines+markers', name='ops', line=dict(color='black')),
    row=1, col=3
)

# Update layout
fig.update_layout(
    title_text='Performance Metrics Analysis',
    height=500,
    width=1200,
    showlegend=True
)

# Update axes labels
fig.update_xaxes(title_text='Filter Size', row=1, col=1)
fig.update_xaxes(title_text='Filter Size', row=1, col=2)
fig.update_xaxes(title_text='Filter Size', row=1, col=3)

fig.update_yaxes(title_text='Number of Coefficients', row=1, col=1)
fig.update_yaxes(title_text='Memory Usage', row=1, col=2)
fig.update_yaxes(title_text='Number of Operations', row=1, col=3)

# Show the figure
fig.show()
import dash
import os
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.stats import beta
import dash_bootstrap_components as dbc


# Function to estimate the probability that Beta(a1, b1) > Beta(a2, b2)
def estimate_probability(a1, b1, a2, b2, num_samples=1000):
    samples_beta1 = beta.rvs(a1, b1, size=num_samples)
    samples_beta2 = beta.rvs(a2, b2, size=num_samples)
    prob = np.mean(samples_beta1 > samples_beta2)
    return prob


# Function to plot Beta distributions
def create_beta_plot(a1, b1, a2, b2):
    x = np.linspace(0, 1, 1000)
    beta1_pdf = beta.pdf(x, a1, b1)
    beta2_pdf = beta.pdf(x, a2, b2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=beta1_pdf, mode='lines', name=f'Beta({a1}, {b1})', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=beta2_pdf, mode='lines', name=f'Beta({a2}, {b2})', line=dict(color='red')))

    fig.update_layout(title="Beta Distributions", xaxis_title="x", yaxis_title="Density", showlegend=True)
    return fig


# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Input Beta Distribution Parameters", style={'textAlign': 'center'}),

            # Input fields for Beta(a1, b1)
            dbc.Row([
                dbc.Col(dbc.Label("Alpha 1 (a1)", style={'fontSize': '20px'}), width=6),
                dbc.Col(dcc.Input(id='a1', type='number', value=2, min=0.1, step=0.1, className='form-control',
                                  style={'width': '100px'}), width=6),
            ], justify="center", className='mb-3'),

            dbc.Row([
                dbc.Col(dbc.Label("Beta 1 (b1)", style={'fontSize': '20px'}), width=6),
                dbc.Col(dcc.Input(id='b1', type='number', value=5, min=0.1, step=0.1, className='form-control',
                                  style={'width': '100px'}), width=6),
            ], justify="center", className='mb-3'),

            # Input fields for Beta(a2, b2)
            dbc.Row([
                dbc.Col(dbc.Label("Alpha 2 (a2)", style={'fontSize': '20px'}), width=6),
                dbc.Col(dcc.Input(id='a2', type='number', value=2, min=0.1, step=0.1, className='form-control',
                                  style={'width': '100px'}), width=6),
            ], justify="center", className='mb-3'),

            dbc.Row([
                dbc.Col(dbc.Label("Beta 2 (b2)", style={'fontSize': '20px'}), width=6),
                dbc.Col(dcc.Input(id='b2', type='number', value=5, min=0.1, step=0.1, className='form-control',
                                  style={'width': '100px'}), width=6),
            ], justify="center", className='mb-3'),

            # Button to update plot and probability
            dbc.Row([
                dbc.Col(dbc.Button("Update", id='update-button', color='primary', className='mt-3',
                                   style={'fontSize': '18px'}), width=12)
            ], justify="center"),
        ], width=3),

        # Plot area and probability output
        dbc.Col([
            dcc.Graph(id='beta-plot'),
            html.H4(id='probability-output', className='mt-3', style={'textAlign': 'center', 'fontSize': '18px'}),
        ], width=9),
    ])
], fluid=True)


# Callback to update the plot and estimated probability
@app.callback(
    [Output('beta-plot', 'figure'),
     Output('probability-output', 'children')],
    [Input('a1', 'value'), Input('b1', 'value'),
     Input('a2', 'value'), Input('b2', 'value'),
     Input('update-button', 'n_clicks')]
)
def update_output(a1, b1, a2, b2, n_clicks):
    if n_clicks is None:
        # Default plot when the app starts
        return create_beta_plot(2, 5, 2, 5), "Estimated Probability: ---"

    # Update the plot
    figure = create_beta_plot(a1, b1, a2, b2)

    # Estimate probability
    prob = estimate_probability(a1, b1, a2, b2)
    prob_text = f"Estimated Probability that Beta({a1}, {b1}) > Beta({a2}, {b2}): {prob:.4f}"

    return figure, prob_text


# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Render dynamically sets the port
    app.run_server(host='0.0.0.0', port=port)


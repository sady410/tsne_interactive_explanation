import json

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

import numpy as np
from sklearn import datasets

from components import tsne_param_component, scatter_plot_card, features_ranking_card, overview_card

from tsne_functions import compute_tsne, create_plot_tsne_embedding
from explainer_functions import compute_all_gradients

iris = datasets.load_iris()
diabetes = datasets.load_diabetes()

def prepare_data(selected_dataset):
    if selected_dataset == 'iris':
        return iris.data, iris.target, iris.feature_names
    elif selected_dataset == 'diabetes':
        return diabetes.data, diabetes.target, diabetes.feature_names
    
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


######################################################
#################  LAYOUT  ###########################
######################################################


app.layout = dbc.Container([
    dcc.Store(id='tsne-data'),
    dcc.Store(id='gradients-data'),
    dbc.Row([
        dbc.Col([
            tsne_param_component,
            overview_card
        ], width=3, style={"display": "flex", "flex-direction": "column", "height": "100vh"}),
        dbc.Col([
            scatter_plot_card
        ], width=6),
        dbc.Col([
            features_ranking_card
        ], width=3)
    ], style={"display": "flex"}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Instances Info", className="card-header"),
                dbc.CardBody([
                    html.Div(id='instances-info')
                ])
            ])
        ], width=12)
    ])
], className="mt-1")


#########################################################
#################  CALLBACKS  ###########################
#########################################################


@app.callback(
    Output('tsne-data', 'data'),
    [Input('run-tsne-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')],
    [State('perplexity-input', 'value')],
    [State('max-iterations-input', 'value')]
)
def run_tsne(n_clicks, selected_datasets, perplexity, max_iter):
    if n_clicks > 0:
        if len(selected_datasets) == 0:
            return json.dumps({})

        X, targets, feature_names = prepare_data(selected_datasets)

        Y, P, Q, sigma = compute_tsne(X, no_dims=2, perplexity=perplexity, max_iter=max_iter)
    
        X_list = X.tolist()
        targets = targets.tolist()
        Y_list = Y.tolist()
        P_list = P.tolist()
        Q_list = Q.tolist()
        sigma_list = sigma.tolist()

        fig = create_plot_tsne_embedding(X, Y, targets)
 
        return json.dumps({'X' : X_list, 'labels' : targets, 'feature_names' : feature_names, 'embedding': Y_list, 'P': P_list, 'Q': Q_list, 'sigma': sigma_list, 'figure': fig.to_json()})  # Convert to JSON string
    else:
        return json.dumps({})  # Return empty JSON string
    
@app.callback(
    Output('gradients-data', 'data'),
    [Input('explain-button', 'n_clicks')],
    [State('tsne-data', 'data')]
)
def compute_gradients(n_clicks, tsne_data):
    if n_clicks > 0:
        if not tsne_data:
            return "Please run t-SNE first."

        tsne_data = json.loads(tsne_data)
        X = tsne_data.get('X')  # Retrieve X data
        labels = tsne_data.get('labels')  # Retrieve labels
        Y = tsne_data.get('embedding')  # Retrieve embedding
        P = tsne_data.get('P')  # Retrieve P values
        Q = tsne_data.get('Q')  # Retrieve Q values
        sigma = tsne_data.get('sigma')  # Retrieve sigma values

        gradients = compute_all_gradients(X, Y, P, Q, sigma)

        # TODO: store these gradients

        return json.dumps({})
    else:
        return ""

@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('tsne-data', 'data')]
)
def update_scatter_plot(tsne_data):
    if tsne_data:
        tsne_data = json.loads(tsne_data)
        figure_json = tsne_data.get('figure')
        if figure_json:
            return json.loads(figure_json)
    # If no data or figure is available, return an empty figure or None
    return {}

@app.callback(
    Output('overview-plot', 'figure'),
    [Input('tsne-data', 'data')]
)
def update_overview_plot(tsne_data):
    if tsne_data:
        tsne_data = json.loads(tsne_data)
        figure_json = tsne_data.get('figure')
        if figure_json:
            return json.loads(figure_json)
    # If no data or figure is available, return an empty figure or None
    return {}



###################################################
#################  RUN  ###########################
###################################################


if __name__ == '__main__':
    app.run_server(debug=True)

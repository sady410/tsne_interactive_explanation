import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn import datasets

from components import *
from explainer_functions import (compute_all_gradients,
                                 create_feature_importance_ranking_plot)
from tsne_functions import compute_tsne, create_plot_tsne_embedding

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


app.layout = html.Div([
    dcc.Store(id='tsne-data'),
    html.Div([
        html.Div([
            html.Div([
                tsne_param_component,
                overview_card
            ], className="d-flex flex-column  justify-content-between"),
            html.Div([
                scatter_plot_card
            ], className="flex-grow-1 max-content"),
            html.Div([
                features_ranking_card
            ], className=""),
        ], className="d-flex justify-content-center"),

        html.Div([

            html.Div([
                instances_info
            ], className=""),

        ], className="d-flex justify-content-center "), # TODO

    ], className="flex-grow-1 max-width-container ")
], className="d-flex justify-content-center")


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

        Y, P, Q, sigma = compute_tsne(
            X, no_dims=2, perplexity=perplexity, max_iter=max_iter)

        X_list = X.tolist()
        targets = targets.tolist()
        Y_list = Y.tolist()
        P_list = P.tolist()
        Q_list = Q.tolist()
        sigma_list = sigma.tolist()

        gradients = compute_all_gradients(X, Y, P, Q, sigma)

        tsne_scatter_plot = create_plot_tsne_embedding(X, Y, targets)
        explanation_bar_plot = create_feature_importance_ranking_plot(gradients, feature_names)

        # Convert to JSON string
        return json.dumps({'X': X_list, 'labels': targets, 'feature_names': feature_names, 
                           'embedding': Y_list, 'P': P_list, 'Q': Q_list, 'sigma': sigma_list, 
                           'gradients': gradients.tolist(),'tsne_scatterplot': tsne_scatter_plot.to_json(), 
                           'explanation_barplot': explanation_bar_plot.to_json()})
    else:
        return json.dumps({})  # Return empty JSON string

@app.callback(
        Output('explanation-barplot', 'figure'),
        [Input('tsne-data', 'data')]
)
def update_explanation_bar_plot(gradients_data):
    if gradients_data:
        gradients_data = json.loads(gradients_data)
        figure_json = gradients_data.get('explanation_barplot')
        if figure_json:
            return json.loads(figure_json)
    return {}

# @app.callback(
#         Output('average-feature-distribution-plot', 'figure'),
#         [Input('tsne-data', 'data')]
# )
# def update_average_feature_distribution_plot(tsne_data):
#     if tsne_data:
#         tsne_data = json.loads(tsne_data)
#         return create_average_feature_distribution_plot()
#     return {}
    

@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('tsne-data', 'data')]
)
def update_scatter_plot(tsne_data):
    if tsne_data:
        tsne_data = json.loads(tsne_data)
        figure_json = tsne_data.get('tsne_scatterplot')
        if figure_json:
            return json.loads(figure_json)
    return {}


@app.callback(
    Output('overview-plot', 'figure'),
    [Input('tsne-data', 'data')]
)
def update_overview_plot(tsne_data):
    if tsne_data:
        tsne_data = json.loads(tsne_data)
        figure_json = tsne_data.get('tsne_scatterplot')
        if figure_json:
            return json.loads(figure_json)
    # If no data or figure is available, return an empty figure or None
    return {}


###################################################
#################  RUN  ###########################
###################################################

if __name__ == '__main__':
    app.run_server(debug=True)

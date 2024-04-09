import dash
import dash_bootstrap_components as dbc
from components import *
from dash.dependencies import Input, Output, State
import json


def layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
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

            ], className="d-flex justify-content-center "),  # TODO

        ], className="flex-grow-1 max-width-container ")
    ], className="d-flex justify-content-center")


@dash.callback(
    Output('tsne-plot', 'figure'),
    [Input('tsne-data', 'data')]
)
def update_scatter_plot(tsne_data):
    if tsne_data is not None and tsne_data:
        tsne_data = json.loads(tsne_data)
        figure_json = tsne_data.get('tsne_scatterplot')
        if figure_json:
            return json.loads(figure_json)
    # If no data or figure is available, return an empty figure or None
    return {}


@dash.callback(
    Output('overview-plot', 'figure'),
    [Input('tsne-data', 'data')]
)
def update_overview_plot(tsne_data):
    if tsne_data is not None and tsne_data:
        tsne_data = json.loads(tsne_data)
        figure_json = tsne_data.get('tsne_scatterplot')
        if figure_json:
            return json.loads(figure_json)
    # If no data or figure is available, return an empty figure or None
    return {}


@dash.callback(
    Output('explanation-barplot', 'figure'),
    [Input('tsne-data', 'data')]
)
def update_explanation_bar_plot(tsne_data):
    if tsne_data is not None and tsne_data:
        gradients_data = json.loads(tsne_data)
        figure_json = gradients_data.get('explanation_barplot')
        if figure_json:
            return json.loads(figure_json)
    return {}

dash.register_page(__name__)
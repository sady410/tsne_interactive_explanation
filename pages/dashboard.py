import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State

from plots import (create_average_feature_distribution_plot,
                   create_combined_gradients_plot,
                   create_feature_importance_ranking_plot)


def overview_card():
    return dcc.Graph(
        id='overview-plot',
        config={'displayModeBar': False,
                'staticPlot': True},
        className="overview-plot ")


def scatter_plot_card():
    return dcc.Graph(
        id='tsne-plot',
        config={'displaylogo': False,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['zoomIn2d',
                                           'zoomOut2d',
                                           'toImage',
                                           'autoScale2d',
                                           'select2d']
                },
        className="main-plot")


def explanation_plot_card():
    return dcc.Graph(id='explanation-barplot',
                     config={'displayModeBar': False},
                     className="explanation-barplot",
                     responsive=True)


def feature_distribution_plot():
    return dcc.Graph(id='feature-distribution-plot',
                     config={'displayModeBar': False},
                     className="feature-distribution-plot",
                     responsive=True)


def layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Overview Plot", className="section-title"),
                overview_card(),
            ]),
            html.Div([
                html.Div("Feature distribution", className="section-title"),
                feature_distribution_plot()
            ], className="h-100")
        ], className="sub-container-1"),
        html.Div([
            html.Div("t-SNE", className="section-title"),
            scatter_plot_card()
        ], className="sub-container-2"),
        html.Div([
            html.Div("Explanation Plot", className="section-title"),
            explanation_plot_card()
        ], className="sub-container-3"),
    ], className="main-container")


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
    return {}


@dash.callback(
    Output('overview-plot', 'figure'),
    [State('tsne-data', 'data')],
    [Input('tsne-plot', 'relayoutData')],
    prevent_initial_call=True
)
def update_overview_plot(tsne_data, relayout_data):
    fig = {}
    if tsne_data is not None and tsne_data:
        tsne_data = json.loads(tsne_data)
        figure_json = tsne_data.get('tsne_scatterplot')
        if figure_json:
            fig = json.loads(figure_json)

    if relayout_data is not None and 'xaxis.range[0]' in relayout_data:
        x0 = relayout_data['xaxis.range[0]']
        x1 = relayout_data['xaxis.range[1]']
        y0 = relayout_data['yaxis.range[0]']
        y1 = relayout_data['yaxis.range[1]']

        # Create a new figure if it doesn't exist
        if 'layout' not in fig:
            fig['layout'] = {}

        # Retrieve existing figure layout or create a new layout if it doesn't exist
        layout = fig['layout']
        shapes = layout.get('shapes', [])

        # Add rectangle shape
        shapes.append({
            'type': 'rect',
            'x0': x0,
            'x1': x1,
            'y0': y0,
            'y1': y1,
            'line': {
                'color': 'rgba(128, 0, 128, 1)',
                'width': 3
            }
        })

        # Update layout with new shapes
        layout['shapes'] = shapes
        fig['layout'] = layout

    return fig


@dash.callback(
    Output('explanation-barplot', 'figure'),
    [Input('tsne-data', 'data'),
     Input('tsne-plot', 'selectedData'),
     Input('tsne-plot', 'clickData')]
)
def update_explanation_bar_plot(tsne_data, selected_data, click_data):
    ctx = dash.callback_context
    triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if tsne_data is not None and tsne_data:
        gradients_data = json.loads(tsne_data)
        gradients = gradients_data.get('gradients')
        feature_names = gradients_data.get('feature_names')
        gradients = np.array(gradients)

        if triggered_component_id == 'tsne-plot':
            if selected_data is not None and selected_data['points'] != []:
                selected_indices = [point['customdata'][0]
                                    for point in selected_data['points']]
                return create_combined_gradients_plot(gradients, feature_names, selected_indices[0])
            elif click_data is not None:
                selected_indices = [click_data['points'][0]['customdata'][0]]
                return create_combined_gradients_plot(gradients, feature_names, selected_indices[0])

        return create_feature_importance_ranking_plot(gradients, feature_names)
    return {}


@dash.callback(
    Output('feature-distribution-plot', 'figure'),
    [Input('tsne-data', 'data'),
     Input('tsne-plot', 'selectedData'),
     Input('tsne-plot', 'clickData')]
)
def update_feature_distribution_plot(tsne_data, selected_data, click_data):
    ctx = dash.callback_context
    triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if tsne_data is not None and tsne_data:
        data = json.loads(tsne_data)
        X = data.get('X')
        feature_names = data.get('feature_names')
        X = np.array(X)

        selected_indices = [i for i in range(X.shape[0])]

        if triggered_component_id == 'tsne-plot':
            if selected_data is not None and selected_data['points'] != []:
                selected_indices = [point['customdata'][0]
                                    for point in selected_data['points']]
            elif click_data is not None:
                selected_indices = [click_data['points'][0]['customdata'][0]]

        return create_average_feature_distribution_plot(feature_names, X, selected_indices)

    return {}


dash.register_page(__name__)

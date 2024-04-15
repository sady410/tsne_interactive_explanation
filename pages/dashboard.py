import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State

from plots import (create_average_feature_distribution_plot,
                   create_combined_gradients_plot,
                   create_feature_importance_ranking_plot,
                   create_plot_tsne_embedding)


def overview_card():
    return html.Div(
        [
            html.Div(
                [
                    dcc.Loading(
                        id="loading-overview-plot",
                        type="circle",
                        className="overview-plot ",
                        children=[
                            dcc.Graph(
                                id='overview-plot', config={'displayModeBar': False, 'staticPlot': True}, className="overview-plot ")
                        ]
                    )

                ]
            )
        ], className=""
    )


def scatter_plot_card():
    return html.Div(
        [
            html.Div(
                [
                    dcc.Loading(
                        id="loading-tsne-plot",
                        type="circle",
                        className="main-plot",
                        children=[
                            dcc.Graph(
                                id='tsne-plot',  config={'displaylogo': False, 'displayModeBar': True, 'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'toImage', 'autoScale2d', 'select2d']}, className="main-plot")
                        ]
                    )
                ],
            )
        ], className="",
    )


def explanation_plot_card():
    return html.Div(
        [
            html.Div("Explanation Plot", className="section-title"), # TODO: Change title explanation is global
            html.Div(
                [
                    dcc.Graph(id='explanation-barplot', 
                              config={'displayModeBar': False}, className="explanation-plot")
                ]
            )
        ],
        className="",
    )


def feature_distribution_plot():
    return html.Div([
        dbc.Col([
            html.Div([
                html.Div("Feature distribution", className="section-title"),
                html.Div(
                [
                    dcc.Graph(id='feature-distribution-plot',
                              config={'displayModeBar': False}, className="feature-distribution-plot")
                ]
            )
            ])
        ])
    ])


def layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    overview_card()
                ], className="d-flex flex-column  justify-content-between"),
                html.Div([
                    scatter_plot_card()
                ], className="flex-grow-1 max-content"),
                html.Div([
                    explanation_plot_card()
                ], className=""),
            ], className="d-flex justify-content-center"),

            html.Div([

                html.Div([
                    feature_distribution_plot()
                ], className=""),

            ], className="d-flex justify-content-center "),  # TODO

        ], className="flex-grow-1 max-width-container ")
    ], className="d-flex justify-content-center")


@dash.callback(
    Output('tsne-plot', 'figure'),
    [Input('tsne-data', 'data'),
     Input('explanation-barplot', 'clickData')],
    [State('tsne-plot', 'figure')]
)
def update_scatter_plot(tsne_data, click_data, figure):

    ctx = dash.callback_context
    triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if tsne_data is not None and tsne_data:
        tsne_data = json.loads(tsne_data)

        X = np.array(tsne_data.get('X'))
        Y = np.array(tsne_data.get('embedding'))
        targets = np.array(tsne_data.get('labels'))

        fig = create_plot_tsne_embedding(X, Y, targets)

        if triggered_component_id == 'explanation-barplot':
            if click_data is not None:
                print(click_data)
                # TODO: fig draw line
        return fig

    return {}


@dash.callback(
    Output('overview-plot', 'figure'),
    [State('overview-plot', 'figure')],
    [Input('tsne-plot', 'relayoutData'),
     Input('tsne-plot', 'figure')],
    prevent_initial_call=True
)
def update_overview_plot(overview_plot, relayout_data, tsne_plot): # TODO: Unzoom
    print(relayout_data)
    if overview_plot is not None:
        fig = overview_plot
    else:   
        fig = tsne_plot
    
    if relayout_data is not None:
        layout = fig['layout']
        shapes = layout.get('shapes', [])
        
        if 'xaxis.autorange' in relayout_data or 'xaxis.range[0]' in relayout_data:
            shapes = []

        if 'xaxis.range[0]' in relayout_data:
            
            x0 = relayout_data['xaxis.range[0]']
            x1 = relayout_data['xaxis.range[1]']
            y0 = relayout_data['yaxis.range[0]']
            y1 = relayout_data['yaxis.range[1]']

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
                selected_indices = [point['customdata'][0] for point in selected_data['points']]
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
                selected_indices = [point['customdata'][0] for point in selected_data['points']]
            elif click_data is not None:
                selected_indices = [click_data['points'][0]['customdata'][0]]    
        
        return create_average_feature_distribution_plot(feature_names, X, selected_indices)

    return {}

dash.register_page(__name__)

import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

from css_colors_exposed import Color
from plots import (create_average_feature_distribution_plot,
                   create_combined_gradients_plot,
                   create_feature_importance_ranking_plot,
                   create_plot_tsne_embedding)


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
    return dcc.Graph(id='explanation-barplot', clear_on_unhover=True,
                     config={'displayModeBar': False},
                     className="explanation-barplot")


def feature_distribution_plot():
    return dcc.Graph(id='feature-distribution-plot',
                     config={'displayModeBar': False},
                     className="feature-distribution-plot")


def layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Overview Plot", className="section-title"),
                overview_card(),
            ]),
            html.Div("Feature distribution", className="section-title mt-4"),
            feature_distribution_plot()
        ], className="sub-container-1"),
        html.Div([
            scatter_plot_card()
        ], className="sub-container-2"),
        html.Div([
            html.Div("Explanation Plot", className="section-title"),
            html.Div(explanation_plot_card(), className="overflow-scroll")
        ], className="sub-container-3"),
    ], className="main-container")


@dash.callback(
    Output('tsne-plot', 'figure'),
    [Input('tsne-data', 'data'),
     Input('explanation-barplot', 'clickData'),
     Input('tsne-plot', 'selectedData')],
    [State('tsne-plot', 'figure'),
     State('explanation-barplot', 'figure')]
)
def update_scatter_plot(tsne_data, click_data, selected_data, tsne_figure, explanation_figure): # TODO: CAN'T HOVER ON SELECTED DATA
    
    ctx = dash.callback_context
    triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    tsne_data = json.loads(tsne_data)
    Y = np.array(tsne_data.get('embedding'))
    
    if triggered_component_id == 'explanation-barplot' or triggered_component_id == 'tsne-plot':
    # TODO: FIX: There is a bug here, when no selected features vectors are still plotted
        
        fig = tsne_figure
        
        gradients = np.array(tsne_data.get('gradients'))

        layout = fig['layout']
        shapes = []
            
        
        if triggered_component_id == 'explanation-barplot':
            if explanation_figure['data'][0]['marker']['color'][click_data['points'][0]['pointIndex']] == Color.primary.value:
                layout['shapes'] = shapes
                fig['layout'] = layout
                return fig

        if 'selectedpoints' in fig['data'][0]:
            selected_points = []
            for i in range(len(fig['data'])):
                points_idx = fig['data'][i]['selectedpoints']
                selected_points += [fig['data'][i]['customdata'][point_id][0] for point_id in points_idx]
        else:
            selected_points = []
            for i in range(len(fig['data'])):
                selected_points += [i[0] for i in fig['data'][i]['customdata']]

        if click_data is not None:
            coordinates = Y[selected_points]
            gradients = gradients[selected_points]

            feature_id = click_data['points'][0]['pointIndex']

            for i in range(coordinates.shape[0]):
                x0, y0 = coordinates[i]
                x1, y1 = coordinates[i] + gradients[i, :, feature_id]*1 # TODO: DETERMINE SCALING FACTOR
                shapes.append({
                    'type': 'line',
                    'x0': x0,
                    'x1': x1,
                    'y0': y0,
                    'y1': y1,
                    'line': {
                        'color': Color.danger.value,
                        'width': 2
                    }
                })
            # TODO -> Can you update the exaplanation graph to make the selected feature use Color.primary.value? @sady410
            
        layout['shapes'] = shapes
        fig['layout'] = layout

    else:
        
        X = np.array(tsne_data.get('X'))
        targets = np.array(tsne_data.get('labels'))
        dataset_name = tsne_data.get('dataset_name')

        fig = create_plot_tsne_embedding(X, Y, targets, dataset_name)

    return fig

@dash.callback(
    Output('overview-plot', 'figure'),
    [State('overview-plot', 'figure')],
    [Input('tsne-plot', 'relayoutData'),
    Input('tsne-plot', 'figure')],
    prevent_initial_call=True
)
def update_overview_plot(overview_plot, relayout_data, tsne_plot):
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
                    'color': Color.info.value,
                    'width': 2
                }
            })

          
        layout['shapes'] = shapes
        fig['layout'] = layout

    return fig


@dash.callback(
    Output('explanation-barplot', 'figure'),
    [Input('tsne-data', 'data'),
     Input('tsne-plot', 'selectedData'),
     Input('explanation-barplot', 'clickData'),
     State('explanation-barplot', 'figure')]
)
def update_explanation_bar_plot(tsne_data, selected_data, click_data, figure):
    ctx = dash.callback_context
    triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    
    gradients_data = json.loads(tsne_data)
    gradients = gradients_data.get('gradients')
    feature_names = gradients_data.get('feature_names')
    gradients = np.array(gradients)
    fig = figure
    if triggered_component_id == 'tsne-plot':
        if selected_data is not None and selected_data['points'] != []:
            selected_indices = [point['customdata'][0]
                                for point in selected_data['points']]
            colors = fig['data'][0]['marker']['color']
            fig = create_combined_gradients_plot(gradients, feature_names, selected_indices[0])
            
            fig.update_traces(marker=dict(color = colors))
        else:
            colors = fig['data'][0]['marker']['color']
            fig = create_feature_importance_ranking_plot(gradients, feature_names)
            fig.update_traces(marker=dict(color = colors))
    elif triggered_component_id == 'explanation-barplot':
        fig = go.Figure(figure)

        colors = [Color.primaryBorderSubtle.value for i in range(len(fig.data[0]['x']))]
        feature_id = click_data['points'][0]['pointIndex'] 
        
        if fig['data'][0]['marker']['color'][feature_id] != Color.primary.value:
            colors[feature_id] = Color.primary.value
        
        fig.update_traces(marker=dict(color = colors))

    else:
        fig = create_feature_importance_ranking_plot(gradients, feature_names)
    
    return fig


@dash.callback(
    Output('feature-distribution-plot', 'figure'),
    [Input('tsne-data', 'data'),
     Input('tsne-plot', 'selectedData'),
     Input('explanation-barplot', 'hoverData'),
     Input('explanation-barplot', 'clickData')],
    [State('feature-distribution-plot', 'figure')]
)
def update_feature_distribution_plot(tsne_data, selected_data, hover_data, click_data, figure):
    ctx = dash.callback_context
    triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]
    fig = go.Figure(figure)
    if triggered_component_id == 'explanation-barplot': # TODO: FULL BUG
        
        if hover_data is not None:
        
            line_colors = [Color.primaryBorderSubtle.value for i in range(len(fig.data[0]['x']))]
            colors = fig['data'][0]['marker']['color']
            # print(colors)
            feature_id = hover_data['points'][0]['pointIndex'] 
            line_colors[feature_id] = Color.secondary.value

            fig.update_traces(marker=dict(color = colors, line=dict(width=2,
                                        color=line_colors)))

        elif click_data is not None:

            colors = [Color.primaryBorderSubtle.value for i in range(len(fig.data[0]['x']))]
    
            feature_id = click_data['points'][0]['pointIndex'] 
            
            if fig['data'][0]['marker']['color'][feature_id] != Color.primary.value:
                colors[feature_id] = Color.primary.value
                
            fig.update_traces(marker=dict(color = colors))
    else:
        data = json.loads(tsne_data)
        X = data.get('X')
        feature_names = data.get('feature_names')
        X = np.array(X)

        selected_indices = [i for i in range(X.shape[0])]

        if triggered_component_id == 'tsne-plot' or triggered_component_id == 'explanation-barplot':
            if selected_data is not None and selected_data['points'] != []:
                selected_indices = [point['customdata'][0]
                                    for point in selected_data['points']]
        fig  = create_average_feature_distribution_plot(feature_names, X, selected_indices)
                
    return fig


dash.register_page(__name__)

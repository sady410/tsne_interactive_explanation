import json
from turtle import width

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
                html.Div("t-SNE Overview", className="section-title"),
                overview_card(),
            ]),
            html.Div("Global Average Feature Distribution", id="feature-distribution-plot-title", className="section-title mt-4"),
            feature_distribution_plot()
        ], className="sub-container-1"),
        html.Div([
            scatter_plot_card()
        ], className="sub-container-2"),
        html.Div([
            html.Div("Global Feature Importance", id="explanation-barplot-title", className="section-title"),
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
    X = np.array(tsne_data.get('X'))

    fig = tsne_figure
    shapes = []
    
    if fig is None:
        targets = np.array(tsne_data.get('labels'))
        dataset_name = tsne_data.get('dataset_name')
        return create_plot_tsne_embedding(X, Y, targets, dataset_name)
    else:
        if triggered_component_id == 'explanation-barplot':
            layout = fig['layout']
            if len(fig["data"]) == 4:
                fig["data"].pop(3) # remove contour plot
            if explanation_figure['data'][0]['marker']['color'][click_data['points'][0]['pointIndex']] == Color.primary.value: 
                
                layout['shapes'] = shapes
                fig['layout'] = layout

                return fig
            else:
                
                if 'selectedpoints' in fig['data'][0]:
                    selected_points = []
                    for i in range(len(fig['data'])):
                        points_idx = fig['data'][i]['selectedpoints']
                        selected_points += [fig['data'][i]['customdata'][point_id][0] for point_id in points_idx]
                else:
                    selected_points = []
                    for i in range(len(fig['data'])-1):
                        print(i)
                        print(fig['data'][i])
                        selected_points += [i[0] for i in fig['data'][i]['customdata']]

                coordinates = Y[selected_points]
                gradients = np.array(tsne_data.get('gradients'))

                feature_id = click_data['points'][0]['pointIndex']

                for i in range(coordinates.shape[0]):
                    point_id = selected_points[i]
                    x0, y0 = coordinates[i]
                    x1, y1 = coordinates[i] + gradients[point_id, :, feature_id]*1 # TODO: DETERMINE SCALING FACTOR
                    shapes.append({
                        'type': 'line',
                        'x0': x0,
                        'x1': x1,
                        'y0': y0,
                        'y1': y1,
                        'line': {
                            'color': Color.danger.value,
                            'width': 2
                        },
                        'opacity': 0.8
                    })
                # TODO -> Can you update the exaplanation graph to make the selected feature use Color.primary.value? @sady410

                layout['shapes'] = shapes
                fig['layout'] = layout
                
                fig = go.Figure(fig)
                
                fig.add_trace(go.Contour(x=Y[:,0],y=Y[:,1],z=np.array(X[:, feature_id])))

                return fig
        elif triggered_component_id == 'tsne-plot':

            layout = fig['layout']
            
            if Color.primary.value not in explanation_figure['data'][0]['marker']['color']: # no selected feature
                layout['shapes'] = shapes
                fig['layout'] = layout
                return fig
            else:
                if selected_data is None:
                    selected_points = [i for i in range(len(Y))]
                elif selected_data['points'] == []:
                    layout['shapes'] = shapes
                    fig['layout'] = layout
                    return fig
                else:
                    selected_points = [selected_data['points'][i]['customdata'][0] for i in range(len(selected_data['points']))]

                coordinates = Y[selected_points]
                gradients = np.array(tsne_data.get('gradients'))

                feature_id = explanation_figure['data'][0]['marker']['color'].index(Color.primary.value)

                for i in range(coordinates.shape[0]):
                    point_id = selected_points[i]
                    x0, y0 = coordinates[i]
                    x1, y1 = coordinates[i] + gradients[point_id, :, feature_id]*1 # TODO: DETERMINE SCALING FACTOR
                    shapes.append({
                        'type': 'line',
                        'x0': x0,
                        'x1': x1,
                        'y0': y0,
                        'y1': y1,
                        'line': {
                            'color': Color.danger.value,
                            'width': 2
                        },
                        'opacity': 0.8
                    })
                # TODO: Can you update the exaplanation graph to make the selected feature use Color.primary.value? @sady410

                layout['shapes'] = shapes
                fig['layout'] = layout
                return fig
        else:
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
    Output('explanation-barplot-title', 'children'),
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
            title = "Feature Importance for Selected Points"
        else:
            colors = fig['data'][0]['marker']['color']
            fig = create_feature_importance_ranking_plot(gradients, feature_names)
            fig.update_traces(marker=dict(color = colors))
            title = "Global Feature Importance"
    elif triggered_component_id == 'explanation-barplot':
        fig = go.Figure(figure)

        colors = [Color.primaryBorderSubtle.value for i in range(len(fig.data[0]['x']))]
        feature_id = click_data['points'][0]['pointIndex'] 
        
        if fig['data'][0]['marker']['color'][feature_id] != Color.primary.value:
            colors[feature_id] = Color.primary.value
        
        fig.update_traces(marker=dict(color = colors))
        title = "Feature Importance for Selected Points"
    else:
        fig = create_feature_importance_ranking_plot(gradients, feature_names)
        title = "Global Feature Importance"
    
    return fig, title


@dash.callback(
    Output('feature-distribution-plot', 'figure'),
    Output('feature-distribution-plot-title', 'children'),
    [Input('tsne-data', 'data'),
     Input('tsne-plot', 'selectedData'),
     Input('explanation-barplot', 'hoverData'),
     Input('explanation-barplot', 'clickData')],
    [State('feature-distribution-plot', 'figure'),
     State('feature-distribution-plot-title', 'children')]
)
def update_feature_distribution_plot(tsne_data, selected_data, hover_data, click_data, figure, title):

    ctx = dash.callback_context
    triggered_component_id, triggered_event = ctx.triggered[0]['prop_id'].split('.')
    title = title
    if triggered_component_id == 'explanation-barplot':
        fig = go.Figure(figure)
        colors = [Color.primaryBorderSubtle.value for i in range(len(fig.data[0]['x']))]
        line_colors = [Color.primaryBorderSubtle.value for i in range(len(fig.data[0]['x']))]
        if triggered_event == 'hoverData':
            colors = fig['data'][0]['marker']['color']
            if hover_data is not None:
                feature_id = hover_data['points'][0]['pointIndex'] 
                line_colors[feature_id] = Color.primary.value

        elif triggered_event == 'clickData':
            feature_id = click_data['points'][0]['pointIndex'] 
            if fig['data'][0]['marker']['color'][feature_id] != Color.primary.value:
                colors[feature_id] = Color.primary.value
            line_colors[feature_id] = Color.primary.value
            
        
        fig.update_traces(marker=dict(color = colors, line=dict(color=line_colors, width=3)))

    else:
        data = json.loads(tsne_data)
        X = data.get('X')
        feature_names = data.get('feature_names')
        X = np.array(X)

        selected_indices = [i for i in range(X.shape[0])]

        if triggered_component_id == 'tsne-plot' or triggered_component_id == 'explanation-barplot':
            if selected_data is not None and selected_data['points'] != []:
                title = "Average Feature Distribution for Selected Points"
                selected_indices = [point['customdata'][0]
                                    for point in selected_data['points']]
            else:
                title = "Global Average Feature Distribution"
        
        fig  = create_average_feature_distribution_plot(feature_names, X, selected_indices)
        
        if figure is not None:
            fig.update_traces(marker=figure['data'][0]['marker'])
       
    return fig, title


dash.register_page(__name__)

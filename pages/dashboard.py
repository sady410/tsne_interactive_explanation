import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import json
from dash import dcc, html


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
                                id='overview-plot', config={'displayModeBar': False}, className="overview-plot ")
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
                                id='tsne-plot',  config={'displayModeBar': False}, className="main-plot")
                        ]
                    )
                ],
            )
        ], className="",
    )


def features_ranking_card():
    return html.Div(
        [
            html.Div("Features Ranking", className="section-title"),
            html.Div(
                [
                    dcc.Graph(id='explanation-barplot',
                              className="explanation-plot")
                ]
            )
        ],
        className="",
    )


def instances_info():
    return html.Div([
        dbc.Col([
            html.Div([
                html.Div("Instances Info", className="section-title"),
                html.Div(id='instances-info')
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
                    features_ranking_card()
                ], className=""),
            ], className="d-flex justify-content-center"),

            html.Div([

                html.Div([
                    instances_info()
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

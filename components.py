from pydoc import classname

import dash_bootstrap_components as dbc
from dash import dcc, html


overview_card = html.Div(
    [
        html.Div(
            [
                dcc.Loading(
                    id="loading-overview-plot",
                    type="circle",
                    className="overview-plot ",
                    children=[
                        dcc.Graph(id='overview-plot', config={'displayModeBar': False}, className="overview-plot ")
                    ]
                )

            ]
        )
    ], className=""
)

scatter_plot_card = html.Div(
    [
        html.Div(
            [
                dcc.Loading(
                    id="loading-tsne-plot",
                    type="circle",
                    className="main-plot",
                    children=[
                        dcc.Graph(id='tsne-plot',  config={'displayModeBar': False}, className="main-plot")
                    ]
                )
            ],
        )
    ], className="", 
)

features_ranking_card = html.Div(
    [
        html.Div("Features Ranking", className="section-title"),
        html.Div(
            [
                dcc.Graph(id='explanation-barplot', className="explanation-plot")
            ]
        )
    ],
    className="", 
)

instances_info = html.Div([
        dbc.Col([
            html.Div([
                html.Div("Instances Info", className="section-title"),
                    html.Div(id='instances-info')
            ])
        ])
    ])
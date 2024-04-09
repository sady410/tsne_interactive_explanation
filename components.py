from pydoc import classname
import dash_bootstrap_components as dbc
from dash import dcc, html

tsne_param_component = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        dbc.Label("Dataset", className="input-label"),
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[
                                {'label': 'Iris', 'value': 'iris'},
                                {'label': 'Diabetes', 'value': 'diabetes'}
                            ],
                            value=None,
                            multi=False,
                            className=""
                        ),
                    ],
                    className="",
                ),
                html.Hr(),
                html.Div(
                    [
                        dbc.Label("Perplexity", className="input-label"),
                        dbc.Input(id="perplexity-input", placeholder="", type="number", size="sm", value=30),
                    ],
                    className="",
                ),
                html.Div(
                    [
                        dbc.Label("Max Iterations", className="input-label"),
                        dbc.Input(id="max-iterations-input", placeholder="", type="number", size="sm", value=400),
                    ],
                    className="",
                    
                ),
                html.Div(
                    [
                        dbc.Button('Run t-SNE', id='run-tsne-button', n_clicks=0, color='primary', className="w-100"),
                    ],
                    className="mt-3"
                )
            ]
        )
    ], className="parameters-container"
)

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
                dcc.Graph(id='features-ranking-plot', className="explanation-plot")
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
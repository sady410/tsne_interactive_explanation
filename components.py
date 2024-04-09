import dash_bootstrap_components as dbc
from dash import dcc, html

tsne_param_component = html.Div(
    [
        html.Div("Parameters", className=""),
        html.Div(
            [
                html.Div(
                    [
                        dbc.Label("Dataset"),
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
                html.Div(
                    [
                        dbc.Label("Perplexity"),
                        dbc.Input(id="perplexity-input", placeholder="", type="number", size="sm", value=30),
                    ],
                    className="",
                ),
                html.Div(
                    [
                        dbc.Label("Max Iterations"),
                        dbc.Input(id="max-iterations-input", placeholder="", type="number", size="sm", value=400),
                    ],
                    className="",
                    
                ),
                html.Div(
                    [
                        dbc.Button('Run t-SNE', id='run-tsne-button', n_clicks=0, color='primary', className='mr-2'),
                        dbc.Button('Explain', id='explain-button', n_clicks=0, color='secondary', className='mr-2')
                    ],
                    className=""
                )
            ]
        )
    ], className="parameters-container"
)

overview_card = html.Div(
    [
        dbc.Label("Overview"),
        html.Div(
            [
                dcc.Loading(
                    id="loading-overview-plot",
                    type="circle",
                    children=[
                        dcc.Graph(id='overview-plot', config={'displayModeBar': False})
                    ]
                )

            ]
        )
    ], className=""
)

scatter_plot_card = html.Div(
    [
        dbc.Label("Scatter Plot"),
        html.Div(
            [
                dcc.Loading(
                    id="loading-scatter-plot",
                    type="circle",
                    children=[
                        dcc.Graph(id='tsne-plot',  config={'displayModeBar': False})
                    ]
                )
            ],
        )
    ], className="", 
)

features_ranking_card = html.Div(
    [
        dbc.Label("Features Ranking"),
        html.Div(
            [
                dcc.Graph(id='features-ranking-plot')
            ]
        )
    ],
    className="", 
)

instances_info = html.Div([
        dbc.Col([
            html.Div([
                dbc.Label("Instances Info", className="card-header"),
                    html.Div(id='instances-info')
            ])
        ])
    ])
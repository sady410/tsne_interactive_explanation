import dash_bootstrap_components as dbc
from dash import dcc, html

tsne_param_component = dbc.Card(
    [
        dbc.CardHeader("Parameters", className="card-header"),
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Label('Dataset', className="mr-2", style={'font-size': 'small', 'flex': '1'}),
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[
                                {'label': 'Iris', 'value': 'iris'},
                                {'label': 'Diabetes', 'value': 'diabetes'}
                            ],
                            value=None,
                            multi=False,
                            style={'width': '70%', 'font-size': 'small', 'flex': '2'}
                        ),
                    ],
                    className="mb-3",
                    style={'display': 'flex', 'align-items': 'center'}
                ),
                html.Div(
                    [
                        html.Label('Perplexity', className="mr-2", style={'font-size': 'small', 'flex': '1'}),
                        dcc.Input(
                            id='perplexity-input',
                            type='number',
                            value=30,
                            style={'width': '70%', 'font-size': 'small', 'flex': '2'}
                        ),
                    ],
                    className="mb-3",
                    style={'display': 'flex', 'align-items': 'center'}
                ),
                html.Div(
                    [
                        html.Label('Max Iterations', className="mr-2", style={'font-size': 'small', 'flex': '1'}),
                        dcc.Input(
                            id='max-iterations-input',
                            type='number',
                            value=400,
                            style={'width': '70%', 'font-size': 'small', 'flex': '2'}
                        ),
                    ],
                    className="mb-3",
                    style={'display': 'flex', 'align-items': 'center'}
                ),
                html.Div(
                    [
                        dbc.Button('Run t-SNE', id='run-tsne-button', n_clicks=0, color='primary', className='mr-2', style={'font-size': 'small'}),
                        dbc.Button('Explain', id='explain-button', n_clicks=0, color='secondary', className='mr-2', style={'font-size': 'small'})
                    ],
                    className="d-flex justify-content-between"
                )
            ]
        )
    ], className="my-2", style={"flex": "1"}
)

overview_card = dbc.Card(
    [
        dbc.CardHeader("Overview", className="card-header"),
        dbc.CardBody(
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
    ], className="my-2", style={"flex": "1"}
)

scatter_plot_card = dbc.Card(
    [
        dbc.CardHeader("Scatter Plot", className="card-header"),
        dbc.CardBody(
            [
                dcc.Loading(
                    id="loading-scatter-plot",
                    type="circle",
                    children=[
                        dcc.Graph(id='tsne-plot',  config={'displayModeBar': False})
                    ]
                )
            ],
            style={'height': 'calc(100% - 50px)'}  # Subtracting header height
        )
    ], className="my-2", style={"height": "100vh"}
)

features_ranking_card = dbc.Card(
    [
        dbc.CardHeader("Features Ranking", className="card-header"),
        dbc.CardBody(
            [
                dcc.Graph(id='features-ranking-plot')
            ]
        )
    ],
    className="my-2", style={"height": "100vh"}
)
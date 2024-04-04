import dash_bootstrap_components as dbc
from dash import dcc, html

tsne_param_component = dbc.Card(
    [
        dbc.CardHeader("Parameters", className="card-header"),
        dbc.CardBody(
            [
                html.Label('Dataset'),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[
                        {'label': 'Iris', 'value': 'iris'},
                        {'label': 'Diabetes', 'value': 'diabetes'}
                    ],
                    value=None,
                    multi=False
                ),
                html.Label('Perplexity'),
                dcc.Slider(
                    id='perplexity-slider',
                    min=4,
                    max=50,
                    step=1,
                    value=30,
                    marks={i: str(i) for i in range(0, 51, 10)}
                ),
                html.Label('Max Iterations'),
                dcc.Slider(
                    id='max-iterations-slider',
                    min=100,
                    max=1000,
                    step=100,
                    value=400,
                    marks={i: str(i) for i in range(100, 1001, 100)}
                ),
                html.Div(
                    [
                        dbc.Button('Run t-SNE', id='run-tsne-button', n_clicks=0, color='primary', className='mr-2'),
                        dbc.Button('Explain', id='explain-button', n_clicks=0, color='secondary', className='mr-2')
                    ],
                    className="d-flex justify-content-between"
                )
            ]
        )
    ]
)

overview_card = dbc.Card(
    [
        dbc.CardHeader("Overview", className="card-header"),
        dbc.CardBody(
            [
                html.Div(id='overview')
            ]
        )
    ]
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
                        dcc.Graph(id='tsne-plot'),
                    ]
                )
            ],
            style={'height': 'calc(100% - 50px)'}  # Subtracting header height
        )
    ]
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
    className="mb-3"
)
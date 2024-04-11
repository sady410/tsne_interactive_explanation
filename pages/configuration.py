import dash
import dash_bootstrap_components as dbc
import dash
import json
from dash.dependencies import Input, Output, State
from dash import dcc, html

def layout():
    return html.Div([
        tsne_param_component()
    ], className="d-flex justify-content-center")


def tsne_param_component():
    return html.Div(
        [
            html.Div("InsightSNE", className="page-title text-center w-100 my-5"),
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
                            dbc.Input(id="perplexity-input", placeholder="",
                                      type="number", size="sm", value=30),
                        ],
                        className="",
                    ),
                    html.Div(
                        [
                            dbc.Label("Max Iterations",
                                      className="input-label mt-3"),
                            dbc.Input(id="max-iterations-input", placeholder="",
                                      type="number", size="sm", value=400),
                        ],
                        className="",

                    ),
                    html.Hr(),
                    html.Div(
                        [
                            dbc.Button('Visualize', id='run-tsne-button',
                                       n_clicks=0, color='primary', className="w-100"),
                        ],
                        className="mt-3"
                    )
                ], className="card"
            )
        ], className="parameters-container mt-5"
    )


def run_tsne(selected_datasets, perplexity, max_iter):
    from tsne import compute_tsne
    from plots import create_plot_tsne_embedding
    from explainer import compute_all_gradients

    def prepare_data(selected_dataset):
        from sklearn import datasets

        iris = datasets.load_iris()
        diabetes = datasets.load_diabetes()

        if selected_dataset == 'iris':
            return iris.data, iris.target, iris.feature_names
        elif selected_dataset == 'diabetes':
            return diabetes.data, diabetes.target, diabetes.feature_names

    if len(selected_datasets) == 0:
        return json.dumps({})

    X, targets, feature_names = prepare_data(selected_datasets)

    Y, P, Q, sigma = compute_tsne(X, no_dims=2, perplexity=perplexity, max_iter=max_iter)

    X_list = X.tolist()
    targets = targets.tolist()
    Y_list = Y.tolist()
    P_list = P.tolist()
    Q_list = Q.tolist()
    sigma_list = sigma.tolist()

    gradients = compute_all_gradients(X, Y, P, Q, sigma)

    tsne_scatter_plot = create_plot_tsne_embedding(X, Y, targets)

    # Convert to JSON string
    return json.dumps({'X': X_list, 'labels': targets, 'feature_names': feature_names,
                       'embedding': Y_list, 'P': P_list, 'Q': Q_list, 'sigma': sigma_list,
                       'gradients': gradients.tolist(), 'tsne_scatterplot': tsne_scatter_plot.to_json()})


@dash.callback(
    Output('url', 'pathname'),
    Output('tsne-data', 'data'),
    [Input('run-tsne-button', 'n_clicks')],
    [State('url', 'pathname')],
    [State('dataset-dropdown', 'value')],
    [State('perplexity-input', 'value')],
    [State('max-iterations-input', 'value')]

)
def navigate(n_clicks, current_url, selected_datasets, perplexity, max_iter):
    if n_clicks > 0:
        return 'dashboard', run_tsne(selected_datasets, perplexity, max_iter)
    else:
        return current_url, json.dumps({})


dash.register_page(__name__, path='/')
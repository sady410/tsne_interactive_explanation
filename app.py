import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from flask import session
from sklearn import datasets

from components import *
from explainer_functions import compute_all_gradients
from tsne_functions import compute_tsne, create_plot_tsne_embedding

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    dcc.Store(id='tsne-data', storage_type='session'),
    dash.page_container
], className="d-flex justify-content-center")


if __name__ == '__main__':
    app.run_server(debug=True)

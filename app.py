import dash
import dash_bootstrap_components as dbc
from dash import dcc, html


app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    dcc.Store(id='tsne-data', storage_type='session'),
    dash.page_container
])  


if __name__ == '__main__':
    app.run_server(debug=True)

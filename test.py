import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    dbc.Row([
        dbc.Col([
            html.Div("First Column Row 1", className="my-2", style={"background-color": "lightblue", "flex": "1"}),
            html.Div("First Column Row 2", className="my-2", style={"background-color": "lightblue", "flex": "1"})
        ], width=1, style={"display": "flex", "flex-direction": "column", "height": "100vh"}),
        dbc.Col(html.Div("Second Column", className="my-2", style={"background-color": "lightgreen", "height": "100vh"}), width=3),
        dbc.Col(html.Div("Third Column", className="my-2", style={"background-color": "lightcoral", "height": "100vh"}), width=3),
        dbc.Col(html.Div("Fourth Column", className="my-2", style={"background-color": "lightsalmon", "height": "100vh"}), width=5),
    ], style={"display": "flex"}),
)

if __name__ == '__main__':
    app.run_server(debug=True)

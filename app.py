import dash
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc

from layout import create_layout
from callbacks import register_callbacks
from util.parsers import get_data

# Load data once (passed into callbacks)
df, xs, ys = get_data()

def create_app():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    load_figure_template("FLATLY")
    app.title = "Population displacement after earthquakes"
    app.layout = create_layout(xs, ys, df)   # pass initial data
    register_callbacks(app, df)              # attach callbacks
    return app

app = create_app()
server = app.server  # needed for deployment (Gunicorn, etc.)

if __name__ == "__main__":
    app.run(debug=True)

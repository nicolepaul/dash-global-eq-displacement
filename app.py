import dash
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc

from _config import *
from util.parsers import get_data

from layout import main_layout
from callback import register_callbacks, register_tab_callbacks

def create_app():
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,  
    )

    load_figure_template("FLATLY")
    app._favicon = "favicon.ico"
    app.title = "Population displacement after earthquakes"

    df, xs, ys = get_data()

    app.layout = main_layout(xs, ys, df)

    register_callbacks(app, df)
    register_tab_callbacks(app, xs, ys, df)

    return app


# Initialize app and server
app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True)
import dash
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc

from _config import *
from util.parsers import get_data, get_drivers
from util.analysis import transform_variables

from layout import main_layout
from callback import register_callbacks, register_tab_callbacks

def create_app(production=True):
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,  
    )

    load_figure_template("FLATLY")
    app._favicon = "favicon.ico"
    app.title = "Population displacement after earthquakes"

    df, xs, ys, zs = get_data()
    drivers = get_drivers()
    df, drivers = transform_variables(df, drivers)

    app.layout = main_layout(xs, ys, df, drivers)

    register_callbacks(app, df, drivers, production=production)
    register_tab_callbacks(app, xs, ys, zs, drivers)

    return app


# Initialize app and server
app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True)
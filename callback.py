from dash import Input, Output, html
import dash_bootstrap_components as dbc

from layout import layout_damage, layout_drivers, layout_contributors
from callbacks.callbacks_damage import register_callbacks_damage
from callbacks.callbacks_drivers import register_callbacks_drivers
from callbacks.callbacks_contributors import register_callbacks_contributors


def register_callbacks(app, df, drivers, production=True):
    register_callbacks_damage(app, df)
    register_callbacks_drivers(app, df, drivers, production=production)
    register_callbacks_contributors(app)


def register_tab_callbacks(app, xs, ys, drivers):
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab):
        if tab == "tab-damage":
            content = layout_damage(xs, ys)
        elif tab == "tab-drivers":
            content = layout_drivers(ys, drivers)
        elif tab == "tab-contributors":
            content =  layout_contributors()
        else:
            content = html.Div("Tab not found")
        return dbc.Card(
            dbc.CardBody(content),
            className="tab"
        )
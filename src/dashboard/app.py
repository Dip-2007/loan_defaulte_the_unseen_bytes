import dash
import dash_bootstrap_components as dbc
try:
    from layout import create_layout
    from callbacks import register_callbacks
except ImportError:
    from .layout import create_layout
    from .callbacks import register_callbacks

# Initialize app with a dark theme compatible stylesheet and Bootstrap Icons
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP, "/assets/style.css"], suppress_callback_exceptions=True)
app.title = "Pre-Delinquency Engine"

# Set layout
app.layout = create_layout(app)

# Register callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)

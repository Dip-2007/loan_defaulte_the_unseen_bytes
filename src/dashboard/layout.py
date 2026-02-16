from dash import html, dcc
import dash_bootstrap_components as dbc


def _kpi_card(title, value_id, color='#e8eaed', subtitle_id=None, icon=None):
    """Minimal KPI card with optional icon and subtitle."""
    body = [
        html.Div(className="kpi-label", children=title),
        html.Div(id=value_id, className="kpi-value mt-1", style={"color": color}),
    ]
    if subtitle_id:
        body.append(html.Div(id=subtitle_id, className="kpi-delta mt-1"))
    return dbc.Card(dbc.CardBody(body, className="py-3 px-3"), className="glass-card")


def _band_column(band_name, css_class, list_id, btn_id, width=2):
    """Risk band column with header + scrollable customer list."""
    return dbc.Col([
        html.Div(band_name, className=f"band-header {css_class}"),
        html.Div([
            html.Div(id=list_id),
            dbc.Button("Load more", id=btn_id,
                       className="btn-subtle w-100 mt-2",
                       style={"display": "none"})
        ], className="scrollable-list",
           style={"maxHeight": "400px", "overflowY": "auto"})
    ], width=width)


def create_layout(app):
    navbar = dbc.NavbarSimple(
        brand=html.Div([
            html.Div(
                html.I(className="bi bi-shield-check"),
                className="d-flex justify-content-center align-items-center me-2",
                style={
                    "height": "32px", "width": "32px",
                    "borderRadius": "8px",
                    "background": "linear-gradient(135deg, #5b8def, #56ccf2)",
                    "color": "white", "fontSize": "0.9rem"
                }
            ),
            html.Span("Pre-Delinquency Engine",
                       style={"fontWeight": "600", "fontSize": "0.9rem",
                              "letterSpacing": "0.3px"})
        ], className="d-flex align-items-center"),
        brand_href="#",
        color="white",
        dark=False,
        fluid=True,
    )

    return html.Div([
        navbar,
        dbc.Container([

            # ═══════════════════════════════════════════════════
            # ROW 1: Portfolio Overview  (KPIs + Donut)
            # Admin sees the big picture at a glance
            # ═══════════════════════════════════════════════════
            html.Div("Portfolio Overview",
                     className="section-title mt-3 mb-2"),

            dbc.Row([
                # Left: 4 KPI cards in a 2×2 grid
                dbc.Col([
                    dbc.Row([
                        dbc.Col(_kpi_card("Total Customers", "total-customers",
                                         "#e8eaed"), width=6),
                        dbc.Col(_kpi_card("At Risk", "high-risk-count",
                                         "#ef4444", subtitle_id="at-risk-pct"),
                                width=6),
                    ], className="g-3 mb-3"),
                    dbc.Row([
                        dbc.Col(_kpi_card("Potential Exposure", "potential-loss",
                                         "#f59e0b"), width=6),
                        dbc.Col(_kpi_card("Avg Risk Score", "avg-risk-score",
                                         "#5b8def"), width=6),
                    ], className="g-3"),
                ], width=5),

                # Center: Risk distribution donut (the key visual)
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Risk Distribution"),
                        dbc.CardBody(
                            dcc.Graph(id="risk-distribution-chart",
                                      config={"displayModeBar": False},
                                      style={"height": "180px"}),
                            className="p-1"
                        )
                    ], className="glass-card chart-card h-100"),
                    width=3
                ),

                # Right: Score histogram (distribution detail)
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Score Distribution"),
                        dbc.CardBody(
                            dcc.Graph(id="score-histogram",
                                      config={"displayModeBar": False},
                                      style={"height": "180px"}),
                            className="p-1"
                        )
                    ], className="glass-card chart-card h-100"),
                    width=4
                ),
            ], className="g-3"),

            # ═══════════════════════════════════════════════════
            # ROW 2: Analysis  (Employment + Risk Drivers)
            # Admin understands WHY risk exists
            # ═══════════════════════════════════════════════════
            html.Div("Risk Analysis",
                     className="section-title mt-4 mb-2"),

            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Risk by Employment Type"),
                        dbc.CardBody(
                            dcc.Graph(id="employment-bar-chart",
                                      config={"displayModeBar": False},
                                      style={"height": "230px"}),
                            className="p-1"
                        )
                    ], className="glass-card chart-card"), width=6
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Top Risk Drivers"),
                        dbc.CardBody(
                            dcc.Graph(id="shap-importance-chart",
                                      config={"displayModeBar": False},
                                      style={"height": "230px"}),
                            className="p-1"
                        )
                    ], className="glass-card chart-card"), width=6
                ),
            ], className="g-3"),

            # ═══════════════════════════════════════════════════
            # ROW 3: Customer Segmentation  (5 band columns)
            # Admin drills into individual customers
            # ═══════════════════════════════════════════════════
            html.Div("Customer Segmentation",
                     className="section-title mt-4 mb-2"),

            dbc.Row([
                _band_column("Safe",     "band-safe", "safe-risk-list",
                             "btn-load-more-safe", width=2),
                _band_column("Low Risk", "band-low",  "low-risk-list",
                             "btn-load-more-low", width=2),
                _band_column("Moderate", "band-mod",  "moderate-risk-list",
                             "btn-load-more-mod", width=3),
                _band_column("High",     "band-high", "high-risk-list",
                             "btn-load-more-high", width=3),
                _band_column("Critical", "band-crit", "critical-risk-list",
                             "btn-load-more-critical", width=2),
            ], className="g-3 mb-4"),

            # Modal for customer details
            dbc.Modal([
                dbc.ModalHeader(
                    dbc.ModalTitle("Customer Details",
                                  style={"fontSize": "1rem"})),
                dbc.ModalBody(id="modal-content")
            ], id="customer-detail-modal", size="xl", is_open=False),

            # Data trigger
            dcc.Interval(id='initial-load-interval', interval=100,
                         max_intervals=1)

        ], fluid=True, className="px-4 pb-4 dashboard-container")
    ])

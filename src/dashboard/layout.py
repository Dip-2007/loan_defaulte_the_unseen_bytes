from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout(app):
    navbar = dbc.NavbarSimple(
        brand=html.Div([
            html.Div(
                html.I(className="bi bi-shield-fill-exclamation"),
                className="d-flex justify-content-center align-items-center me-2",
                style={
                    "height": "40px",
                    "width": "40px",
                    "borderRadius": "50%",
                    "background": "linear-gradient(135deg, #0dcaf0 0%, #0d6efd 100%)",
                    "color": "white",
                    "fontSize": "1.2rem"
                }
            ),
            html.Span("Pre-Delinquency Intervention Engine", style={"fontWeight": "bold", "letterSpacing": "1px"})
        ], className="d-flex align-items-center"),
        brand_href="#",
        color="#2c3034",  # Lighter than black, dark gray (Material Dark)
        dark=True,
        fluid=True,
        brand_style={'fontSize': '1.1rem'}
    )

    return html.Div([
        navbar,
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total Customers", id="kpi-total-tooltip-target"),
                        dbc.Tooltip("Total number of active customers in the database.", target="kpi-total-tooltip-target", placement="top"),
                        dbc.CardBody([
                            html.Div([
                                html.H3("1,245", id="total-customers", className="mb-0", style={"color": "#2b7de9"}), # Custom blue between dark and light
                                html.Small("Active", className="text-muted")
                            ], className="d-flex flex-column"),
                            
                            html.Div([
                                html.Div([html.Span("●", className="text-success me-1"), html.Span("Low:", className="text-success me-1"), html.Span("0", id="breakdown-low")], className="d-flex align-items-center small"),
                                html.Div([html.Span("●", className="text-warning me-1"), html.Span("Med:", className="text-warning me-1"), html.Span("0", id="breakdown-med")], className="d-flex align-items-center small"),
                                html.Div([html.Span("●", className="text-danger me-1"), html.Span("High:", className="text-danger me-1"), html.Span("0", id="breakdown-high")], className="d-flex align-items-center small"),
                            ], className="border-start border-secondary ps-3 ms-3")
                        ], className="d-flex justify-content-between align-items-center")
                    ], className="glass-card mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("High Risk Count", id="kpi-risk-tooltip-target", style={"cursor": "help"}),
                        dbc.Tooltip("Number of customers with a Risk Score > 600 (Likely to default).", target="kpi-risk-tooltip-target", placement="top"),
                        dbc.CardBody(html.H3("143", id="high-risk-count", className="text-danger"))
                    ], className="glass-card mb-4")
                ], width=4),

                 dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Potential Loss", id="kpi-loss-tooltip-target"),
                         dbc.Tooltip("Estimated total exposure from high-risk accounts.", target="kpi-loss-tooltip-target", placement="top"),
                        dbc.CardBody(html.H3("$1.2M", id="potential-loss", className="text-warning"))
                    ], className="glass-card mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Intervention Success", id="kpi-success-tooltip-target"),
                        dbc.Tooltip("Percentage of high-risk customers recovered after intervention.", target="kpi-success-tooltip-target", placement="top"),
                        dbc.CardBody(html.H3("84%", id="intervention-success", className="text-success"))
                    ], className="glass-card mb-4")
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Distribution"),
                        dbc.CardBody(dcc.Graph(id="risk-distribution-chart", style={"height": "250px"}))
                    ], className="glass-card mb-4 full-height")
                ], width=4),
            ], className="mt-4 mb-2"), 

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                         dbc.CardHeader("Employment Demographics", className="glass-header"),
                         dbc.CardBody(dcc.Graph(id="employment-bar-chart", style={"height": "200px"}))
                    ], className="glass-card mb-4")
                ], width=6),
            ], className="mb-4"),

            # ... (KPI Row Code which is largely unchanged, skipping to Risk List Row)

            # ... 

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Low Risk Customers", className="glass-header text-success glow-border-success mb-4"),
                        dbc.CardBody(
                            html.Div([
                                html.Div(id="low-risk-list"),
                                dbc.Button("See More...", id="btn-load-more-low", color="link", className="text-decoration-none text-success w-100 p-2")
                            ], className="scrollable-list", style={"maxHeight": "400px", "overflowY": "auto"})
                        , className="glass-body-container glow-border-success p-0")
                    ], className="transparent-card full-height")
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Medium Risk Customers", className="glass-header text-warning glow-border-warning mb-4"),
                        dbc.CardBody(
                            html.Div([
                                html.Div(id="medium-risk-list"),
                                dbc.Button("See More...", id="btn-load-more-med", color="link", className="text-decoration-none text-warning w-100 p-2")
                            ], className="scrollable-list", style={"maxHeight": "400px", "overflowY": "auto"})
                        , className="glass-body-container glow-border-warning p-0")
                    ], className="transparent-card full-height")
                ], width=4),
                
                 dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("High Risk Customers", className="glass-header text-danger glow-border-danger mb-4"),
                        dbc.CardBody(
                            html.Div([
                                html.Div(id="high-risk-list"),
                                dbc.Button("See More...", id="btn-load-more-high", color="link", className="text-decoration-none text-danger w-100 p-2")
                            ], className="scrollable-list", style={"maxHeight": "400px", "overflowY": "auto"})
                        , className="glass-body-container glow-border-danger p-0")
                    ], className="transparent-card full-height")
                ], width=4),
            ], className="mb-4"),
            
            # Modal for Customer Details
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Customer Details"), className="glass-header", style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                dbc.ModalBody(id="modal-content", className="glass-body-container", style={"borderRadius": "0 0 10px 10px"})
            ], id="customer-detail-modal", size="xl", is_open=False, style={"backdropFilter": "blur(5px)"}),
            
            # Initialization Trigger
            dcc.Interval(id='initial-load-interval', interval=100, max_intervals=1)
            
        ], fluid=True, className="p-4 dashboard-container")
    ])

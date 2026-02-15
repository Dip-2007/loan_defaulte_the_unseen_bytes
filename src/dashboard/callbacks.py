from dash import Input, Output, State, html, dcc, ALL, ctx, no_update
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
try:
    from data_loader import load_data, get_customer_timeline, get_spending_breakdown
except ImportError:
    from .data_loader import load_data, get_customer_timeline, get_spending_breakdown

def register_callbacks(app):
    df = load_data()
    
    @app.callback(
        [Output("total-customers", "children"),
         Output("high-risk-count", "children"),
         Output("potential-loss", "children"),
         Output("breakdown-low", "children"),
         Output("breakdown-med", "children"),
         Output("breakdown-high", "children"),
         Output("risk-distribution-chart", "figure"),
         Output("employment-bar-chart", "figure")],
        Input("initial-load-interval", "n_intervals")
    )
    def update_dashboard_stats(_):
        print("DEBUG: update_dashboard_stats triggered")
        print(f"DEBUG: df shape: {df.shape}")
        if 'risk_category' in df.columns:
            print(f"DEBUG: risk_category counts:\n{df['risk_category'].value_counts()}")
        else:
            print("DEBUG: risk_category column MISSING")
            
        print(f"DEBUG: employment_type info: {'Present' if 'employment_type' in df.columns else 'Missing'}")
        
        try:
            # Calculate stats
            total_customers = len(df)
            
            low_risk_count = len(df[df['risk_category'] == 'Low'])
            med_risk_count = len(df[df['risk_category'] == 'Medium'])
            high_risk_count = len(df[df['risk_category'] == 'High'])
            
            # Estimate potential loss (Mock calculation: sum of loan_amount for high risk)
            # Using a fixed average if loan_amount not present
            if 'loan_amount' in df.columns:
                potential_loss = df[df['risk_score'] > 600]['loan_amount'].sum()
            else:
                 potential_loss = high_risk_count * 15000 # Mock average loan
                 
            # Format currency
            loss_formatted = f"${potential_loss:,.0f}"

            # Create Pie Chart
            labels = ['Low Risk', 'Medium Risk', 'High Risk']
            values = [low_risk_count, med_risk_count, high_risk_count]
            colors = ['#00ff9d', '#ffaa00', '#FF0000'] # Success, Original Warning, Pure Red

            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False
            )
            fig_pie.update_traces(textinfo='percent+label')

            # Create Employment Bar Chart
            if 'employment_type' in df.columns:
                emp_counts = df['employment_type'].value_counts().reset_index()
                emp_counts.columns = ['Employment Type', 'Count']
                
                fig_bar = px.bar(emp_counts, x='Employment Type', y='Count', 
                                 color='Employment Type', # distinct colors
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=False,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
            else:
                print("DEBUG: 'employment_type' column missing in df")
                fig_bar = go.Figure()

            return f"{total_customers:,}", f"{high_risk_count}", loss_formatted, f"{low_risk_count}", f"{med_risk_count}", f"{high_risk_count}", fig_pie, fig_bar

        except Exception as e:
            print(f"CRITICAL ERROR in update_dashboard_stats: {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallbacks to avoid blank screen
            return "Error", "Error", "Error", "0", "0", "0", go.Figure(), go.Figure()
    
    @app.callback(
        [Output("low-risk-list", "children"),
         Output("medium-risk-list", "children"),
         Output("high-risk-list", "children"),
         Output("btn-load-more-low", "style"),
         Output("btn-load-more-med", "style"),
         Output("btn-load-more-high", "style")],
        [Input("initial-load-interval", "n_intervals"),
         Input("btn-load-more-low", "n_clicks"),
         Input("btn-load-more-med", "n_clicks"),
         Input("btn-load-more-high", "n_clicks")]
    )
    def update_risk_lists(_, n_low, n_med, n_high):
        print("DEBUG: update_risk_lists triggered")
        try:
            # Default n_clicks is None
            limit_low = ((n_low or 0) + 1) * 20
            limit_med = ((n_med or 0) + 1) * 20
            limit_high = ((n_high or 0) + 1) * 20

            def generate_list(category, text_color, limit):
                full_df = df[df['risk_category'] == category]
                total_count = len(full_df)
                filtered_df = full_df.head(limit)
                
                items = []
                for i, (_, row) in enumerate(filtered_df.iterrows(), 1):
                    customer_name = row.get('name', row['customer_id'])
                    customer_id = row['customer_id']
                    items.append(
                        html.Div([
                            html.Span(f"{i}. {customer_name}", style={'color': '#ffffff', 'textShadow': '0 0 5px rgba(255, 255, 255, 0.5)'}, className="fw-bold"),
                            html.Span(f" Score: {int(row['risk_score'])}", className="ms-2 text-muted small"),
                            html.Hr(className="my-1 border-secondary")
                        ], className="p-2 customer-list-item",
                           id={'type': 'customer-item', 'index': customer_id},
                           n_clicks=0,
                           style={"cursor": "pointer", "transition": "background 0.2s"}) 
                    )
                
                # Determine button visibility
                btn_style = {'display': 'block'} if total_count > limit else {'display': 'none'}
                
                return items, btn_style

            low_items, low_btn_style = generate_list('Low', 'text-success', limit_low)
            med_items, med_btn_style = generate_list('Medium', 'text-warning', limit_med)
            high_items, high_btn_style = generate_list('High', 'text-danger', limit_high)

            return low_items, med_items, high_items, low_btn_style, med_btn_style, high_btn_style

        except Exception as e:
            print(f"CRITICAL ERROR in update_risk_lists: {e}")
            import traceback
            traceback.print_exc()
            return [], [], [], {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    @app.callback(
        [Output("customer-detail-modal", "is_open"),
         Output("modal-content", "children")],
        [Input({'type': 'customer-item', 'index': ALL}, 'n_clicks')],
        [State("customer-detail-modal", "is_open")]
    )
    def toggle_modal(n_clicks, is_open):
        try:
            print(f"DEBUG: toggle_modal triggered. Triggered: {ctx.triggered}")
            if not ctx.triggered:
                return False, no_update
            
            button_id = ctx.triggered_id
            print(f"DEBUG: Triggered ID: {button_id}")
            
            if not button_id or (isinstance(button_id, dict) and button_id.get('type') != 'customer-item'):
                return False, no_update
                
            customer_id = button_id['index']
            print(f"DEBUG: Customer ID clicked: {customer_id}, Type: {type(customer_id)}")
            
            # Robust DataFrame Lookup
            mask = df['customer_id'] == customer_id
            if not mask.any():
                print(f"DEBUG: Direct match failed. Trying type conversion.")
                # Try converting to string if it's an int, or vice versa
                if isinstance(customer_id, int):
                    mask = df['customer_id'] == str(customer_id)
                elif isinstance(customer_id, str) and customer_id.isdigit():
                    mask = df['customer_id'] == int(customer_id)
            
            if not mask.any():
                print(f"ERROR: Customer ID {customer_id} not found in DataFrame.")
                return is_open, no_update # Keep current state

            customer_row = df[mask].iloc[0]
            customer_name = customer_row.get('name', customer_id)
            customer_risk = customer_row['risk_score']

            # Timeline
            timeline_df = get_customer_timeline(customer_id)
            fig_timeline = px.line(timeline_df, x='date', y='balance', title="Account Balance History")
            fig_timeline.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            fig_timeline.update_traces(line_color='#2b7de9', line_width=3)
            
            # Spending
            spending_df = get_spending_breakdown(customer_id)
            fig_spending = px.bar(spending_df, x='category', y='amount', title="Monthly Spending")
            fig_spending.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            fig_spending.update_traces(marker_color='#ff007f')

            # Risk Factors
            risk_factors = []
            if customer_risk > 700:
                risk_factors.append(html.Span("Likely Default", className="badge bg-danger p-2 me-2"))
            if customer_risk > 500:
                risk_factors.append(html.Span("Salary Delayed", className="badge bg-warning text-dark p-2 me-2"))
            risk_factors.append(html.Span("High Utilization", className="badge bg-info p-2 me-2"))
            
            modal_content = html.Div([
                html.H4(f"Analysis for {customer_name}", className="text-white mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Financial Stress Timeline"),
                            dbc.CardBody(dcc.Graph(figure=fig_timeline))
                        ], className="glass-card mb-4")
                    ], width=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Spending Habits"),
                            dbc.CardBody(dcc.Graph(figure=fig_spending))
                        ], className="glass-card mb-4")
                    ], width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H5("Detected Risk Factors", className="mb-3"),
                        html.Div(risk_factors, className="d-flex flex-wrap mb-4")
                    ], width=6),
                    dbc.Col([
                        html.H5("Recommended Actions", className="mb-3"),
                        html.Div([
                            dbc.Button("Offer Payment Holiday", color="success", className="me-2"),
                            dbc.Button("Contact Customer", color="info", className="me-2"),
                            dbc.Button("Flag for Review", color="warning")
                        ])
                    ], width=6)
                ])
            ])
            
            return True, modal_content

        except Exception as e:
            print(f"CRITICAL ERROR in toggle_modal: {e}")
            import traceback
            traceback.print_exc()
            return is_open, no_update

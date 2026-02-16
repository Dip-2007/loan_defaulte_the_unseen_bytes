from dash import Input, Output, State, html, dcc, ALL, ctx, no_update
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
try:
    from data_loader import load_data, load_feature_importance, get_customer_timeline, get_spending_breakdown
except ImportError:
    from .data_loader import load_data, load_feature_importance, get_customer_timeline, get_spending_breakdown

CUST_ID_COL = 'LoanID'

# Design palette — muted, elegant
BAND_COLORS = {
    'SAFE':      '#10b981',
    'LOW RISK':  '#4a7cde',
    'MODERATE':  '#e8930c',
    'HIGH RISK': '#ea7830',
    'CRITICAL':  '#dc3545',
}

def _layout(**overrides):
    """Build chart layout by merging base defaults with per-chart overrides."""
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#5f6477', size=11),
        margin=dict(t=8, b=30, l=40, r=16),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=False,
                   tickfont=dict(size=10, color='#5f6477')),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=False,
                   tickfont=dict(size=10, color='#5f6477')),
        showlegend=False,
        hoverlabel=dict(bgcolor='#ffffff', font_color='#1a1d2e',
                        bordercolor='#e0e0e0'),
    )
    base.update(overrides)
    return base


def _empty_fig():
    fig = go.Figure()
    fig.update_layout(**_layout())
    return fig


def register_callbacks(app):
    df = load_data()
    feat_imp = load_feature_importance()

    # ================================================================
    # KPI + Charts
    # ================================================================
    @app.callback(
        [Output("total-customers", "children"),
         Output("high-risk-count", "children"),
         Output("at-risk-pct", "children"),
         Output("potential-loss", "children"),
         Output("avg-risk-score", "children"),
         Output("risk-distribution-chart", "figure"),
         Output("score-histogram", "figure"),
         Output("employment-bar-chart", "figure"),
         Output("shap-importance-chart", "figure")],
        Input("initial-load-interval", "n_intervals")
    )
    def update_dashboard_stats(_):
        try:
            total = len(df)

            counts = {}
            for band in BAND_COLORS:
                counts[band] = len(df[df['risk_category'] == band])

            at_risk = counts.get('MODERATE', 0) + counts.get('HIGH RISK', 0) + counts.get('CRITICAL', 0)
            at_risk_pct = f"{at_risk / max(total, 1) * 100:.1f}% of portfolio"

            # Potential exposure
            if 'LoanAmount' in df.columns:
                mask = df['risk_category'].isin(['MODERATE', 'HIGH RISK', 'CRITICAL'])
                loss = df.loc[mask, 'LoanAmount'].sum()
            else:
                loss = at_risk * 15000

            if loss >= 1e7:
                loss_str = f"₹{loss / 1e7:.1f}Cr"
            elif loss >= 1e5:
                loss_str = f"₹{loss / 1e5:.1f}L"
            else:
                loss_str = f"₹{loss:,.0f}"

            # Avg risk score
            avg_score = df['risk_score'].mean() if 'risk_score' in df.columns else 0

            # ── Donut Chart ──────────────────────────────────
            labels = list(BAND_COLORS.keys())
            values = [counts.get(b, 0) for b in labels]
            colors = [BAND_COLORS[b] for b in labels]

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.65,
                marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
                textinfo='none', hoverinfo='label+value+percent',
                sort=False,
            )])
            fig_donut.update_layout(**_layout(margin=dict(t=8, b=8, l=8, r=8)))
            # Add center text
            fig_donut.add_annotation(
                text=f"<b>{total:,}</b><br><span style='font-size:9px;color:#9ca3b4'>total</span>",
                showarrow=False, font=dict(size=16, color='#1a1d2e')
            )

            # ── Score Histogram ──────────────────────────────
            fig_hist = go.Figure()
            if 'risk_score' in df.columns:
                # Color each bar by risk band
                bin_edges = [0, 25, 45, 60, 75, 100]
                band_names = ['SAFE', 'LOW RISK', 'MODERATE', 'HIGH RISK', 'CRITICAL']
                band_colors = [BAND_COLORS[b] for b in band_names]

                for i in range(len(bin_edges) - 1):
                    mask = (df['risk_score'] >= bin_edges[i]) & (df['risk_score'] <= bin_edges[i + 1])
                    subset = df.loc[mask, 'risk_score']
                    if len(subset) > 0:
                        fig_hist.add_trace(go.Histogram(
                            x=subset, name=band_names[i],
                            marker_color=band_colors[i],
                            opacity=0.85,
                            xbins=dict(start=bin_edges[i], end=bin_edges[i + 1], size=5),
                            hovertemplate=f"{band_names[i]}<br>Score: %{{x}}<br>Count: %{{y}}<extra></extra>"
                        ))
            fig_hist.update_layout(**_layout(
                barmode='stack', xaxis_title=None, yaxis_title=None, bargap=0.08,
            ))

            # ── Employment Bar ───────────────────────────────
            fig_emp = _empty_fig()
            if 'employment_type' in df.columns:
                # Stacked bar by risk band
                emp_types = df['employment_type'].unique()
                for band in ['CRITICAL', 'HIGH RISK', 'MODERATE', 'LOW RISK', 'SAFE']:
                    band_df = df[df['risk_category'] == band]
                    emp_counts = band_df['employment_type'].value_counts()
                    fig_emp.add_trace(go.Bar(
                        x=[emp_counts.get(e, 0) for e in emp_types],
                        y=emp_types, name=band,
                        orientation='h',
                        marker_color=BAND_COLORS[band],
                        opacity=0.85,
                        hovertemplate=f"{band}<br>%{{y}}: %{{x}}<extra></extra>"
                    ))
                fig_emp.update_layout(**_layout(barmode='stack'))

            # ── SHAP Feature Importance ──────────────────────
            fig_shap = _empty_fig()
            if not feat_imp.empty:
                top = feat_imp.head(8).iloc[::-1]
                imp_col = 'mean_abs_shap' if 'mean_abs_shap' in top.columns else 'importance'
                max_val = top[imp_col].max()
                normalized = top[imp_col] / max_val if max_val > 0 else top[imp_col]

                fig_shap = go.Figure(go.Bar(
                    x=top[imp_col], y=top['feature'],
                    orientation='h',
                    marker=dict(
                        color=normalized,
                        colorscale=[[0, '#b3d4fc'], [0.5, '#4a7cde'], [1, '#38a9d4']],
                    ),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>"
                ))
                fig_shap.update_layout(**_layout(
                    yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10)),
                ))

            return (
                f"{total:,}",
                f"{at_risk}",
                at_risk_pct,
                loss_str,
                f"{avg_score:.1f}",
                fig_donut,
                fig_hist,
                fig_emp,
                fig_shap,
            )
        except Exception as e:
            print(f"ERROR in update_dashboard_stats: {e}")
            import traceback; traceback.print_exc()
            empty = _empty_fig()
            return ("—", "—", "", "—", "—", empty, empty, empty, empty)

    # ================================================================
    # Risk Band Lists
    # ================================================================
    @app.callback(
        [Output("safe-risk-list", "children"),
         Output("low-risk-list", "children"),
         Output("moderate-risk-list", "children"),
         Output("high-risk-list", "children"),
         Output("critical-risk-list", "children"),
         Output("btn-load-more-safe", "style"),
         Output("btn-load-more-low", "style"),
         Output("btn-load-more-mod", "style"),
         Output("btn-load-more-high", "style"),
         Output("btn-load-more-critical", "style")],
        [Input("initial-load-interval", "n_intervals"),
         Input("btn-load-more-safe", "n_clicks"),
         Input("btn-load-more-low", "n_clicks"),
         Input("btn-load-more-mod", "n_clicks"),
         Input("btn-load-more-high", "n_clicks"),
         Input("btn-load-more-critical", "n_clicks")]
    )
    def update_risk_lists(_, n_safe, n_low, n_mod, n_high, n_crit):
        try:
            limits = {
                'SAFE':      ((n_safe or 0) + 1) * 20,
                'LOW RISK':  ((n_low or 0) + 1) * 20,
                'MODERATE':  ((n_mod or 0) + 1) * 20,
                'HIGH RISK': ((n_high or 0) + 1) * 20,
                'CRITICAL':  ((n_crit or 0) + 1) * 20,
            }

            def make_list(category, color, limit):
                full = df[df['risk_category'] == category]
                if 'risk_score' in full.columns:
                    full = full.sort_values('risk_score', ascending=False)
                total_count = len(full)
                subset = full.head(limit)

                items = []
                for _, row in subset.iterrows():
                    name = row.get('name', row[CUST_ID_COL])
                    cid = row[CUST_ID_COL]
                    score = row.get('risk_score', 0)
                    items.append(
                        html.Div([
                            html.Span(str(name), className="cust-name"),
                            html.Span(f"{score:.0f}",
                                      className="cust-score",
                                      style={"color": color,
                                             "background": f"{color}15"})
                        ], className="customer-list-item d-flex justify-content-between align-items-center",
                           id={'type': 'customer-item', 'index': cid},
                           n_clicks=0)
                    )

                btn = {'display': 'block'} if total_count > limit else {'display': 'none'}
                return items, btn

            results = []
            for band in ['SAFE', 'LOW RISK', 'MODERATE', 'HIGH RISK', 'CRITICAL']:
                items, btn = make_list(band, BAND_COLORS[band], limits[band])
                results.append(items)
                results.append(btn)

            return (results[0], results[2], results[4], results[6], results[8],
                    results[1], results[3], results[5], results[7], results[9])

        except Exception as e:
            print(f"ERROR in update_risk_lists: {e}")
            import traceback; traceback.print_exc()
            hide = {'display': 'none'}
            return ([], [], [], [], [], hide, hide, hide, hide, hide)

    # ================================================================
    # Customer Detail Modal
    # ================================================================
    @app.callback(
        [Output("customer-detail-modal", "is_open"),
         Output("modal-content", "children")],
        [Input({'type': 'customer-item', 'index': ALL}, 'n_clicks')],
        [State("customer-detail-modal", "is_open")]
    )
    def toggle_modal(n_clicks, is_open):
        try:
            if not ctx.triggered:
                return False, no_update

            button_id = ctx.triggered_id
            if not button_id or (isinstance(button_id, dict) and
                                 button_id.get('type') != 'customer-item'):
                return False, no_update

            customer_id = button_id['index']

            mask = df[CUST_ID_COL] == customer_id
            if not mask.any():
                if isinstance(customer_id, int):
                    mask = df[CUST_ID_COL] == str(customer_id)
                elif isinstance(customer_id, str) and customer_id.isdigit():
                    mask = df[CUST_ID_COL] == int(customer_id)
            if not mask.any():
                return is_open, no_update

            row = df[mask].iloc[0]
            name = row.get('name', customer_id)
            score = row.get('risk_score', 0)
            band = row.get('risk_category', 'UNKNOWN')
            color = BAND_COLORS.get(band, '#5b8def')

            # ── Timeline chart ───────────────────────────────
            timeline_df = get_customer_timeline(customer_id)
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=timeline_df['date'], y=timeline_df['balance'],
                mode='lines', fill='tozeroy',
                line=dict(color='#4a7cde', width=1.5),
                fillcolor='rgba(74, 124, 222, 0.08)',
                hovertemplate="₹%{y:,.0f}<br>%{x|%b %d}<extra></extra>"
            ))
            fig_tl.update_layout(**_layout(margin=dict(t=8, b=30, l=50, r=16)))

            # ── Spending chart ───────────────────────────────
            spending_df = get_spending_breakdown(customer_id)
            spending_df = spending_df.sort_values('amount', ascending=True)
            fig_sp = go.Figure(go.Bar(
                x=spending_df['amount'], y=spending_df['category'],
                orientation='h',
                marker=dict(
                    color=spending_df['amount'],
                    colorscale=[[0, '#b3d4fc'], [1, '#dc3545']],
                ),
                hovertemplate="₹%{x:,.0f}<extra></extra>"
            ))
            fig_sp.update_layout(**_layout(margin=dict(t=8, b=20, l=80, r=16)))

            # ── Risk gauge ───────────────────────────────────
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(size=10, color='#5f6477')),
                    bar=dict(color=color),
                    bgcolor='#f5f6fa',
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 25], color='rgba(52,211,153,0.15)'),
                        dict(range=[25, 45], color='rgba(91,141,239,0.15)'),
                        dict(range=[45, 60], color='rgba(245,158,11,0.15)'),
                        dict(range=[60, 75], color='rgba(251,146,60,0.15)'),
                        dict(range=[75, 100], color='rgba(239,68,68,0.15)'),
                    ],
                ),
                number=dict(font=dict(size=28, color='#1a1d2e')),
            ))
            fig_gauge.update_layout(**_layout(
                margin=dict(t=30, b=8, l=30, r=30), height=160,
            ))

            # ── Metrics ──────────────────────────────────────
            metrics_data = [
                ('Income', 'Income', '₹{:,.0f}'),
                ('Loan', 'LoanAmount', '₹{:,.0f}'),
                ('Credit Score', 'CreditScore', '{:.0f}'),
                ('DTI', 'DTIRatio', '{:.1f}%'),
                ('On-time Rate', 'ontime_payment_rate_12m', '{:.0%}'),
                ('Savings', 'savings_rate', '{:.1f}%'),
            ]
            metric_items = []
            for label, col, fmt in metrics_data:
                if col in row.index and pd.notna(row[col]):
                    try:
                        val = fmt.format(row[col])
                    except (ValueError, TypeError):
                        val = str(row[col])
                    metric_items.append(
                        html.Div([
                            html.Div(label, style={"fontSize": "0.7rem", "color": "#9ca3b4"}),
                    html.Div(val, style={"fontSize": "0.95rem", "fontWeight": "600",
                                          "color": "#1a1d2e"}),
                        ], className="mb-2")
                    )

            # ── Intervention ─────────────────────────────────
            interventions = {
                'SAFE': ("No action needed", "success"),
                'LOW RISK': ("Send wellness tips within 48h", "info"),
                'MODERATE': ("Proactive outreach — offer restructuring", "warning"),
                'HIGH RISK': ("Urgent call — restructuring + holiday", "danger"),
                'CRITICAL': ("Immediate intervention — emergency loan", "danger"),
            }
            action_text, action_color = interventions.get(band, ("Monitor", "secondary"))

            # ── Why This Score? ──────────────────────────────
            # Evaluate key risk factors from the customer's data
            risk_factors = []

            def _add_factor(label, col, thresholds, higher_is_worse=True, fmt='{:.1f}'):
                """Add a risk factor with severity color.
                thresholds: (good, fair, warning) — values at each boundary.
                """
                if col not in row.index or pd.isna(row[col]):
                    return
                val = row[col]
                try:
                    val_str = fmt.format(val)
                except (ValueError, TypeError):
                    val_str = str(val)

                good_t, fair_t, warn_t = thresholds
                if higher_is_worse:
                    if val <= good_t:
                        severity, sev_color, sev_label = 0, '#10b981', 'Good'
                    elif val <= fair_t:
                        severity, sev_color, sev_label = 1, '#e8930c', 'Fair'
                    elif val <= warn_t:
                        severity, sev_color, sev_label = 2, '#ea7830', 'Warning'
                    else:
                        severity, sev_color, sev_label = 3, '#dc3545', 'Critical'
                else:
                    if val >= good_t:
                        severity, sev_color, sev_label = 0, '#10b981', 'Good'
                    elif val >= fair_t:
                        severity, sev_color, sev_label = 1, '#e8930c', 'Fair'
                    elif val >= warn_t:
                        severity, sev_color, sev_label = 2, '#ea7830', 'Warning'
                    else:
                        severity, sev_color, sev_label = 3, '#dc3545', 'Critical'

                bar_pct = min(max((severity + 1) / 4 * 100, 15), 100)

                risk_factors.append((severity, html.Div([
                    html.Div([
                        html.Span(label, style={"fontSize": "0.78rem", "color": "#5f6477",
                                                "fontWeight": "500"}),
                        html.Span([
                            html.Span(val_str, style={"fontWeight": "600", "color": "#1a1d2e",
                                                       "marginRight": "6px"}),
                            html.Span(sev_label, style={"fontSize": "0.65rem", "color": sev_color,
                                                         "fontWeight": "600",
                                                         "padding": "1px 6px",
                                                         "borderRadius": "4px",
                                                         "background": f"{sev_color}15",
                                                         "border": f"1px solid {sev_color}30"})
                        ])
                    ], className="d-flex justify-content-between align-items-center"),
                    html.Div(
                        html.Div(style={"width": f"{bar_pct}%", "height": "100%",
                                        "borderRadius": "2px", "background": sev_color,
                                        "transition": "width 0.5s ease"}),
                        style={"height": "4px", "borderRadius": "2px",
                               "background": "rgba(0,0,0,0.06)", "marginTop": "4px"}
                    )
                ], className="mb-2")))

            # Key risk signals
            _add_factor("DTI Ratio", "DTIRatio", (30, 45, 60), higher_is_worse=True, fmt='{:.1f}%')
            _add_factor("Computed DTI", "computed_dti", (35, 50, 70), higher_is_worse=True, fmt='{:.1f}%')
            _add_factor("Credit Score", "CreditScore", (750, 650, 550), higher_is_worse=False, fmt='{:.0f}')
            _add_factor("On-time Payments", "ontime_payment_rate_12m", (0.95, 0.85, 0.70), higher_is_worse=False, fmt='{:.0%}')
            _add_factor("Savings Rate", "savings_rate", (15, 8, 3), higher_is_worse=False, fmt='{:.1f}%')
            _add_factor("Credit Utilization", "credit_utilization_ratio", (30, 60, 80), higher_is_worse=True, fmt='{:.0f}%')
            _add_factor("Total Debt / Assets", "total_debt_to_assets", (0.4, 0.7, 0.9), higher_is_worse=True, fmt='{:.2f}')
            _add_factor("Behavioral Risk", "behavioral_risk_score", (30, 50, 70), higher_is_worse=True, fmt='{:.0f}')
            _add_factor("Peer Default Rate", "peer_default_rate", (0.05, 0.15, 0.30), higher_is_worse=True, fmt='{:.1%}')
            _add_factor("Financial Stress", "financial_stress_score", (30, 50, 70), higher_is_worse=True, fmt='{:.0f}')
            _add_factor("Employer Risk", "employer_risk_multiplier", (1.2, 1.5, 2.0), higher_is_worse=True, fmt='{:.2f}x')
            _add_factor("Income Stability", "isi", (70, 50, 30), higher_is_worse=False, fmt='{:.0f}')

            # Sort by severity (worst first) and take top factors
            risk_factors.sort(key=lambda x: x[0], reverse=True)
            factor_items = [item for _, item in risk_factors[:8]]

            # Build "Why This Score?" section
            why_section = html.Div([
                html.Div("Why This Score?",
                         style={"fontSize": "0.75rem", "fontWeight": "600", "color": "#9ca3b4",
                                "textTransform": "uppercase", "letterSpacing": "0.5px",
                                "marginBottom": "8px",
                                "borderLeft": "3px solid #4a7cde",
                                "paddingLeft": "8px"}),
                html.Div(factor_items if factor_items else [
                    html.Div("No detailed factor data available",
                             style={"color": "#9ca3b4", "fontSize": "0.8rem"})
                ])
            ], style={"background": "rgba(255,255,255,0.3)", "borderRadius": "8px",
                       "padding": "12px", "border": "1px solid rgba(0,0,0,0.04)"})

            modal_content = html.Div([
                # Header
                html.Div([
                    html.H5(name, className="mb-0",
                            style={"fontWeight": "600", "color": "#1a1d2e"}),
                    html.Span(band, className="risk-badge ms-3",
                              style={"background": f"{color}20", "color": color,
                                     "border": f"1px solid {color}40"})
                ], className="d-flex align-items-center mb-3"),

                dbc.Row([
                    # Left: gauge + metrics
                    dbc.Col([
                        dcc.Graph(figure=fig_gauge, config={"displayModeBar": False},
                                  style={"height": "160px"}),
                        html.Div(metric_items, className="mt-2")
                    ], width=3),

                    # Center: Why This Score + Timeline
                    dbc.Col([
                        why_section,
                        html.Div("Balance History",
                                 className="mt-3",
                                 style={"fontSize": "0.75rem", "color": "#9ca3b4",
                                        "marginBottom": "4px", "fontWeight": "500"}),
                        dcc.Graph(figure=fig_tl, config={"displayModeBar": False},
                                  style={"height": "160px"}),
                    ], width=5),

                    # Right: spending
                    dbc.Col([
                        html.Div("Monthly Spending",
                                 style={"fontSize": "0.75rem", "color": "#9ca3b4",
                                        "marginBottom": "4px", "fontWeight": "500"}),
                        dcc.Graph(figure=fig_sp, config={"displayModeBar": False},
                                  style={"height": "200px"}),
                    ], width=4),
                ]),

                # Action
                dbc.Alert(action_text, color=action_color,
                          className="mt-3 mb-0",
                          style={"fontSize": "0.85rem", "borderRadius": "8px"})
            ])

            return True, modal_content

        except Exception as e:
            print(f"ERROR in toggle_modal: {e}")
            import traceback; traceback.print_exc()
            return is_open, no_update

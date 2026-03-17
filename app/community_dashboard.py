"""
MindGuard Community Dashboard (v3 - Commercial Edition)
======================================================
Secure, multi-tenant portal for organizations to view their 
aggregate emotional wellness trends, process anonymous batch 
responses, and manage their system settings.
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go

import gradio as gr

# Suppress excessive warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the Database and Analyzer
from model.emotional_analyzer import EmotionalAnalyzer
from db.database import MindGuardDB

ANALYZER = EmotionalAnalyzer()
CFG = ANALYZER.config
DB = MindGuardDB(CFG.get('database', {}).get('path', 'data/mindguard.db'))

COMMUNITY_NAME = CFG.get('community', {}).get('name', 'MindGuard')
DISCLAIMER = CFG.get('community', {}).get('ethical_disclaimer', 'Screening Tool Only.')

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

CSS = """
.disclaimer-box {
    background-color: #FFF3E0;
    border-left: 6px solid #FF9800;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 4px;
    font-size: 0.95em;
}
.stat-card {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
}
.stat-value {
    font-size: 2.5em;
    font-weight: 700;
    margin: 10px 0;
}
.stat-label {
    text-transform: uppercase;
    font-size: 0.8em;
    letter-spacing: 1px;
    color: #64748b;
}
.risk-critical { color: #D32F2F; }
.risk-warning { color: #FF9800; }
.risk-mild { color: #FFC107; }
.risk-none { color: #4CAF50; }

/* Hide footer */
footer {display: none !important;}
"""

# ============================================================
# Core Logic Processors
# ============================================================

def perform_login(api_key):
    """Authenticate organization via API Key."""
    if not api_key:
        return "❌ Please enter an API key.", gr.update(visible=False), None
        
    org_info = DB.validate_api_key(api_key)
    if not org_info:
        return "❌ Invalid or revoked API Key.", gr.update(visible=False), None
        
    org_id = org_info['org_id']
    org_name = org_info['org_name']
    
    welcome_msg = f"### ✅ Welcome, {org_name} ({org_info['tier'].upper()} Plan)\n*Your dashboard is ready.*"
    
    return welcome_msg, gr.update(visible=True), org_id

def load_trend_history(org_id):
    """Load the historical trend dashboard for an org."""
    if not org_id:
        return "<p>Not authenticated.</p>", None, None
        
    org = DB.get_organization(org_id)
    tier = org.get('tier', 'free') if org else 'free'
    if tier == 'free':
        return "<p><i>Trend history is only available on Starter plans and above.</i></p>", None, None
        
    history = DB.get_trend_history(org_id, days=30)
    if not history:
        html = f"""
        <div style='padding:20px; text-align:center; background:#f8fafc; border-radius:8px;'>
            <h3>No Trend Data Yet</h3>
            <p>Process a batch of texts and enable 'Save Snapshot' to build your trend history.</p>
        </div>
        """
        return html, None, None
        
    # Build timeline dataframe
    df = pd.DataFrame([{
        'Date': h['snapshot_date'],
        'Health Score': h['health_score'],
        'Critical Flags': h['critical_flags'],
        'Responses': h['total_analyzed']
    } for h in history])
    
    # Overview HTML
    latest = df.iloc[-1]
    health_color = "#4CAF50" if latest['Health Score'] > 75 else ("#FF9800" if latest['Health Score'] > 50 else "#D32F2F")
    
    html = f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div class="stat-card" style="flex: 1; min-width: 200px;">
            <div class="stat-label">System Status (30d)</div>
            <div class="stat-value" style="font-size: 1.5em; margin-top:20px;">✅ ACTIVE</div>
        </div>
        <div class="stat-card" style="flex: 1; min-width: 200px; border-top: 4px solid {health_color};">
            <div class="stat-label">Current Health Score</div>
            <div class="stat-value" style="color: {health_color};">{latest['Health Score']}/100</div>
        </div>
        <div class="stat-card" style="flex: 1; min-width: 200px; border-top: 4px solid #D32F2F;">
            <div class="stat-label">Total Critical Flags (30d)</div>
            <div class="stat-value risk-critical">{int(df['Critical Flags'].sum())}</div>
        </div>
    </div>
    """
    
    # Health Score Timeline Plot
    fig_health = px.line(df, x='Date', y='Health Score', title="Community Health Trend", markers=True)
    fig_health.update_yaxes(range=[0, 100])
    fig_health.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    # Responses Timeline
    fig_vol = px.bar(df, x='Date', y='Responses', title="Survey Participation Volume")
    fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    return html, fig_health, fig_vol

def process_batch_texts(text_blob: str, save_snapshot: bool, org_id: str):
    """Process a batch of texts and optionally record it to the DB."""
    if not text_blob or not org_id:
        return "Missing data or unauthenticated.", None, None, None
        
    separator = CFG.get('batch', {}).get('separator', '---')
    if separator in text_blob:
        texts = [t.strip() for t in text_blob.split(separator) if t.strip()]
    else:
        texts = [t.strip() for t in text_blob.split('\n\n') if t.strip()]
        
    if not texts:
        return "No usable text blocks found.", None, None, None
        
    # Check rate limit
    org = DB.get_organization(org_id)
    tier = org.get('tier', 'free') if org else 'free'
    tier_limits = CFG.get('pricing', {}).get('tiers', {}).get(tier, {})
    max_batch = tier_limits.get('max_batch_size', 25)
    
    if len(texts) > max_batch:
        return f"❌ Batch too large ({len(texts)}). Your tier ({tier}) limits batches to {max_batch}.", None, None, None
        
    res = ANALYZER.analyze_batch(texts)
    
    if "error" in res:
        return f"Error: {res['error']}", None, None, None
        
    summ = res['summary']
    
    # Save to Database if requested
    DB.log_usage(org_id, endpoint="/analyze/batch_ui", count=len(texts))
    if save_snapshot:
        DB.save_trend_snapshot(
            org_id=org_id,
            total_analyzed=summ['total_analyzed'],
            health_score=summ['overall_health_score'],
            critical_flags=summ['critical_flags_generated'],
            severity_distribution=res['severity_distribution'],
            avg_dimensions=res['average_emotional_dimensions']
        )
    
    # 1. Build Top-Level Stats HTML
    health_color = "#4CAF50" if summ['overall_health_score'] > 75 else ("#FF9800" if summ['overall_health_score'] > 50 else "#D32F2F")
    
    stats_html = f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div class="stat-card" style="flex: 1; min-width: 200px;">
            <div class="stat-label">Total Responses</div>
            <div class="stat-value">{summ['total_analyzed']}</div>
        </div>
        <div class="stat-card" style="flex: 1; min-width: 200px; border-top: 4px solid {health_color};">
            <div class="stat-label">Batch Health Score</div>
            <div class="stat-value" style="color: {health_color};">{summ['overall_health_score']}/100</div>
        </div>
        <div class="stat-card" style="flex: 1; min-width: 200px; border-top: 4px solid #D32F2F;">
            <div class="stat-label">Critical Flags</div>
            <div class="stat-value risk-critical">{summ['critical_flags_generated']}</div>
            <div style="font-size: 0.8em; color: #666;">Requires Immediate Review</div>
        </div>
    </div>
    """
    
    # 2. Plots
    # Severity Donut Chart
    sev_counts = res['severity_distribution']
    colors = {"Severe Crisis": "#D32F2F", "Moderate": "#FF9800", "Mild": "#FFC107", "No Concern": "#4CAF50"}
    labels_sev = list(sev_counts.keys())
    values_sev = list(sev_counts.values())
    marker_colors = [colors.get(l, '#888') for l in labels_sev]
    
    fig_sev = go.Figure(data=[go.Pie(labels=labels_sev, values=values_sev, hole=.4, marker_colors=marker_colors)])
    fig_sev.update_layout(title="Severity Distribution", paper_bgcolor='rgba(0,0,0,0)')
    
    # Dimensions Bar Chart (Averages)
    avg_dims = res['average_emotional_dimensions']
    dim_labels = [ANALYZER.dimensions.get(k, {}).get('label', k) for k in avg_dims.keys()]
    dim_values = list(avg_dims.values())
    dim_colors = [ANALYZER.dimensions.get(k, {}).get('color', '#888') for k in avg_dims.keys()]
    
    fig_dim = px.bar(
        x=dim_labels, 
        y=dim_values,
        color=dim_labels,
        color_discrete_sequence=dim_colors,
        labels={'x': 'Emotional Dimension', 'y': 'Average Intensity'},
        title="Community Average Emotional Profile"
    )
    fig_dim.update_layout(yaxis_range=[0, 100], showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # Table MD
    table_md = "| Severity Level | Count | Action Required |\n|---|---|---|\n"
    for level, count in res['severity_distribution'].items():
        if level == "Severe Crisis": action = "IMMEDIATE REVIEW"
        elif level == "Moderate": action = "Follow-up (48h)"
        elif level == "Mild": action = "Monitor"
        else: action = "None"
        table_md += f"| **{level}** | {count} | {action} |\n"
        
    return stats_html, fig_sev, fig_dim, table_md

# ============================================================
# Gradio Application Definition
# ============================================================

def create_dashboard():
    with gr.Blocks(title=COMMUNITY_NAME, theme=THEME, css=CSS) as app:
        
        org_id_state = gr.State("") # Holds the authenticated org ID
        
        # HEADER
        with gr.Row():
            gr.Markdown(f"# {COMMUNITY_NAME}\n### {CFG.get('community', {}).get('tagline', '')}")
            
        gr.Markdown(f"**Disclaimer:** {DISCLAIMER}", elem_classes=["disclaimer-box"])
        
        # LOGIN SECTION
        with gr.Group(elem_id="login_section") as login_section:
            gr.Markdown("### 🔒 Organization Login")
            gr.Markdown("Please enter your organization's API Key to access your dashboard. (Check `app/admin_dashboard.py` if testing locally).")
            with gr.Row():
                api_key_input = gr.Textbox(label="API Key", type="password", placeholder="mg_...")
                login_btn = gr.Button("Authenticate", variant="primary")
            login_err = gr.Markdown()
        
        # DASHBOARD CONTENT (HIDDEN UNTIL AUTH)
        with gr.Group(visible=False) as main_dashboard:
            
            with gr.Tabs():
            
                # --- TAB 1: RUN ANALYSIS ---
                with gr.Tab("📝 Run Batch Analysis"):
                    gr.Markdown("Copy/paste anonymous responses from Google Forms or SurveyMonkey here.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_input = gr.Textbox(
                                label="Paste Anonymous Check-ins",
                                placeholder="Paste texts here.\nSeparate distinct responses with an empty line.",
                                lines=12
                            )
                            save_toggle = gr.Checkbox(label="💾 Save summary to Monthly Trend History", value=True)
                            run_batch_btn = gr.Button("🔍 Process Responses", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            batch_stats_html = gr.HTML("<p style='color:#666; margin-top:20px;'>Awaiting input...</p>")
                            
                    with gr.Row():
                        batch_plot_sev = gr.Plot()
                        batch_plot_dim = gr.Plot()
                        
                    with gr.Row():
                        batch_table_md = gr.Markdown()
                        
                # --- TAB 2: TREND HISTORY ---
                with gr.Tab("📈 Trend History"):
                    gr.Markdown("View how your community's emotional wellness shifts over time.")
                    refresh_trends_btn = gr.Button("🔄 Refresh Data")
                    
                    trend_html = gr.HTML()
                    with gr.Row():
                        trend_health_plot = gr.Plot()
                        trend_vol_plot = gr.Plot()

        # EVENT BINDINGS
        login_btn.click(
            fn=perform_login,
            inputs=[api_key_input],
            outputs=[login_err, main_dashboard, org_id_state]
        )
        
        run_batch_btn.click(
            fn=process_batch_texts,
            inputs=[batch_input, save_toggle, org_id_state],
            outputs=[batch_stats_html, batch_plot_sev, batch_plot_dim, batch_table_md]
        )
        
        refresh_trends_btn.click(
            fn=load_trend_history,
            inputs=[org_id_state],
            outputs=[trend_html, trend_health_plot, trend_vol_plot]
        )

    return app

if __name__ == "__main__":
    app = create_dashboard()
    app.launch(server_port=7862, share=False)

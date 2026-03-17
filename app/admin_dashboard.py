"""
MindGuard Admin Control Panel
=============================
Internal dashboard for managing organizations, API keys, pricing tiers, 
and monitoring platform-wide analytics. 
NOT for public exposure.
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import json

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

# Suprress excessive warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.database import MindGuardDB
from model.emotional_analyzer import EmotionalAnalyzer

# Initialize
analyzer = EmotionalAnalyzer()
config = analyzer.config
db = MindGuardDB(config.get('database', {}).get('path', 'data/mindguard.db'))

THEME = gr.themes.Monochrome(
    primary_hue="slate",
    secondary_hue="gray",
    font=[gr.themes.GoogleFont('Inter'), 'system-ui', 'sans-serif'],
)

# ==========================================================
# Helpers & Formatting
# ==========================================================

def get_org_dataframe():
    """Fetch active organizations as a pandas dataframe."""
    orgs = db.list_organizations()
    if not orgs:
        return pd.DataFrame(columns=['org_id', 'name', 'email', 'tier', 'created_at'])
    df = pd.DataFrame(orgs)
    df = df[['org_id', 'name', 'email', 'tier', 'created_at']]
    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%b %d, %Y')
    return df

def get_platform_kpi_html():
    """Generate HTML for top-level platform stats."""
    stats = db.get_platform_stats()
    return f"""
    <div style="display:flex; gap:15px; flex-wrap:wrap; margin-bottom:20px;">
        <div style="flex:1; background:#f8fafc; padding:20px; border-radius:8px; border:1px solid #e2e8f0; text-align:center;">
            <p style="color:#64748b; font-size:0.85em; text-transform:uppercase; margin:0;">Active Orgs</p>
            <h2 style="font-size:2em; font-weight:700; color:#0f172a; margin:10px 0;">{stats['total_organizations']}</h2>
        </div>
        <div style="flex:1; background:#f8fafc; padding:20px; border-radius:8px; border:1px solid #e2e8f0; text-align:center;">
            <p style="color:#64748b; font-size:0.85em; text-transform:uppercase; margin:0;">API Keys</p>
            <h2 style="font-size:2em; font-weight:700; color:#0f172a; margin:10px 0;">{stats['total_api_keys']}</h2>
        </div>
        <div style="flex:1; background:#f0f9ff; padding:20px; border-radius:8px; border:1px solid #bae6fd; text-align:center;">
            <p style="color:#0284c7; font-size:0.85em; text-transform:uppercase; margin:0;">MTD Analyses</p>
            <h2 style="font-size:2em; font-weight:700; color:#0369a1; margin:10px 0;">{stats['monthly_analyses']:,}</h2>
        </div>
        <div style="flex:1; background:#f8fafc; padding:20px; border-radius:8px; border:1px solid #e2e8f0; text-align:center;">
            <p style="color:#64748b; font-size:0.85em; text-transform:uppercase; margin:0;">Snapshots Captured</p>
            <h2 style="font-size:2em; font-weight:700; color:#0f172a; margin:10px 0;">{stats['total_snapshots']:,}</h2>
        </div>
    </div>
    """

def get_tier_chart():
    """Plotly chart of active tiers."""
    stats = db.get_platform_stats()
    tiers = stats['tier_distribution']
    if not tiers:
        return px.pie(names=['No Orgs'], values=[1], title="Plan Distribution")
    
    fig = px.pie(
        names=list(tiers.keys()), 
        values=list(tiers.values()),
        hole=0.5,
        color_discrete_sequence=['#94a3b8', '#3b82f6', '#8b5cf6', '#10b981'],
        title="Active Subscriptions by Tier"
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ==========================================================
# Handlers
# ==========================================================

def create_org_handler(name, email, tier):
    if not name or not email:
        return "❌ Please provide a name and email.", get_org_dataframe(), get_platform_kpi_html(), get_tier_chart()
        
    org_id, api_key = db.create_organization(name, email, tier)
    msg = f"""
    ### ✅ Organization Registered Successfully
    **Name:** {name}
    **Tier:** {tier.title()}
    **Org ID:** `{org_id}`
    **Primary API Key:** `{api_key}`
    
    *⚠️ Store this API key securely. It cannot be retrieved later.*
    """
    return msg, get_org_dataframe(), get_platform_kpi_html(), get_tier_chart()

def view_org_usage(org_id):
    if not org_id:
        return "Please enter an Org ID.", None
        
    org = db.get_organization(org_id)
    if not org:
        return "❌ Organization not found or inactive.", None
        
    stats = db.get_usage_stats(org_id, days=30)
    
    tier_info = config.get('pricing', {}).get('tiers', {}).get(org['tier'], {})
    limit = tier_info.get('max_analyses_per_month', 'Unlimited')
    
    html = f"""
    <div style="border:1px solid #e2e8f0; padding:15px; border-radius:6px; background:white;">
        <h3 style="margin-top:0;">{org['name']} <span style="font-size:0.7em; background:#e2e8f0; padding:3px 8px; border-radius:12px; margin-left:10px;">{org['tier'].upper()} Plan</span></h3>
        <p><b>Email:</b> {org['email']}</p>
        <hr style="border:0; border-top:1px solid #e2e8f0; margin:15px 0;" />
        <p><b>30-Day Total Calls:</b> {stats['total_requests']:,} <i>(Tier Limit: {limit})</i></p>
    </div>
    """
    
    # Plotly usage chart
    dates = list(stats['by_date'].keys())
    counts = list(stats['by_date'].values())
    
    if dates:
        fig = px.bar(x=dates, y=counts, title="API Usage (Last 30 Days)", labels={'x':'Date', 'y':'Requests'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    else:
        fig = px.bar(title="No usage data found (Last 30 Days)")
        
    return html, fig

def generate_key_handler(org_id, key_name):
    if not org_id:
        return "❌ Please enter an Org ID."
        
    org = db.get_organization(org_id)
    if not org:
        return "❌ Organization not found."
        
    new_key = db.create_api_key(org_id, key_name or "Additional Key")
    return f"✅ **New Key Generated for {org['name']}:** `{new_key}`"

# ==========================================================
# Gradio UI Layout
# ==========================================================

def app_layout():
    with gr.Blocks(title="MindGuard Admin", theme=THEME) as app:
        
        with gr.Row():
            gr.Markdown("# 🛡️ MindGuard Admin Control Panel")
            gr.Markdown("*Internal Platform Management*", elem_classes="text-right")
            
        # Top KPIs
        kpi_output = gr.HTML(get_platform_kpi_html())
        
        with gr.Tabs():
            
            # --- TAB 1: Organizations & Onboarding ---
            with gr.Tab("🏢 Organizations"):
                with gr.Row():
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Register New Client")
                        org_name = gr.Textbox(label="Organization Name")
                        org_email = gr.Textbox(label="Billing Email")
                        org_tier = gr.Dropdown(
                            choices=list(config.get('pricing', {}).get('tiers', {}).keys()),
                            value="starter",
                            label="Subscription Tier"
                        )
                        create_btn = gr.Button("Register Organization", variant="primary")
                        create_out = gr.Markdown()
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 Active Organizations")
                        org_table = gr.Dataframe(value=get_org_dataframe(), headers=["Org ID", "Name", "Email", "Tier", "Created"])
                        tier_plot = gr.Plot(value=get_tier_chart())
                        
            # --- TAB 2: Usage & Billing Analytics ---
            with gr.Tab("📈 Billing & Usage"):
                with gr.Row():
                    
                    with gr.Column(scale=1):
                        lookup_id = gr.Textbox(label="Enter Organization ID", placeholder="org_...")
                        lookup_btn = gr.Button("Fetch Usage Profile", variant="secondary")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🔑 API Key Management")
                        key_name = gr.Textbox(label="New Key Description (e.g. 'Typeform Int.')")
                        gen_key_btn = gr.Button("Generate Additional Key")
                        key_out = gr.Markdown()
                        
                        
                    with gr.Column(scale=2):
                        usage_html = gr.HTML("<p style='color:#666;'>Enter an Org ID to load usage stats.</p>")
                        usage_plot = gr.Plot()
                        
            # --- TAB 3: Global System Config ---
            with gr.Tab("⚙️ System Config"):
                gr.Markdown("### 📜 Current Pricing Configuration")
                config_json = gr.JSON(value=config.get('pricing', {}))
                
        # Event bindings
        create_btn.click(
            fn=create_org_handler,
            inputs=[org_name, org_email, org_tier],
            outputs=[create_out, org_table, kpi_output, tier_plot]
        )
        
        lookup_btn.click(
            fn=view_org_usage,
            inputs=[lookup_id],
            outputs=[usage_html, usage_plot]
        )
        
        gen_key_btn.click(
            fn=generate_key_handler,
            inputs=[lookup_id, key_name],
            outputs=[key_out]
        )
        
    return app

if __name__ == "__main__":
    app_layout().launch(server_port=7861, show_error=True)

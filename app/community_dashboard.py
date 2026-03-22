"""
MindGuard Community Dashboard
=============================
A comprehensive web interface for organizations (schools, workplaces, communities) 
to process anonymous emotional wellness check-ins, view aggregate trends, and 
identify high-risk flags without compromising individual privacy.

Ethical Constraints Enforced:
1. No text storage
2. Aggregated batch results only
3. Clear disclaimers
4. No diagnostic claims
"""

import sys
import io
import warnings
import threading
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Suppress excessive warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the analyzer
try:
    from model.emotional_analyzer import EmotionalAnalyzer
    ANALYZER = EmotionalAnalyzer()
except Exception as e:
    print(f"[ERROR] Failed to import EmotionalAnalyzer: {e}")
    ANALYZER = None

# Import Gradio + Plotly
try:
    import gradio as gr
except ImportError:
    print("Install Gradio: pip install gradio")
    sys.exit(1)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARN] Plotly not found. Charts will be text-only. pip install plotly")


# ============================================================
# Configuration & Constants
# ============================================================

CFG = ANALYZER.config if ANALYZER else {}
COMMUNITY_NAME = CFG.get('community', {}).get('name', 'MindGuard Community')
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

.flag-box-critical {
    background-color: #ffebee;
    border: 1px solid #ffcdd2;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 5px;
    color: #b71c1c;
    font-weight: 600;
}

.flag-box-warning {
    background-color: #fff8e1;
    border: 1px solid #ffecb3;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 5px;
    color: #f57f17;
    font-weight: 600;
}

/* Hide footer */
footer {display: none !important;}
"""


# ============================================================
# API Server Threading
# ============================================================
API_THREAD = None

def start_api_server():
    global API_THREAD
    if API_THREAD and API_THREAD.is_alive():
        return "API Server is already running."

    def run_uvicorn():
        import uvicorn
        port = CFG.get('api', {}).get('port', 8000)
        uvicorn.run("api.community_api:app", host="127.0.0.1", port=port, log_level="warning")

    try:
        API_THREAD = threading.Thread(target=run_uvicorn, daemon=True)
        API_THREAD.start()
        port = CFG.get('api', {}).get('port', 8000)
        return f"[OK] MindGuard REST API started on http://127.0.0.1:{port}\n\nEndpoints:\n- POST /api/v1/analyze/single\n- POST /api/v1/analyze/batch"
    except Exception as e:
        return f"[FAIL] Could not start API: {str(e)}"


# ============================================================
# Core Logic: Single Text Analysis
# ============================================================

def process_single_text(text: str):
    """Process a single text and return formatted HTML + plots."""
    if not ANALYZER:
        return "<div class='disclaimer-box'>Analyzer not loaded.</div>", None, None

    if not text or len(text.strip()) < 10:
        return "<div class='disclaimer-box'>Please enter at least 10 characters to analyze.</div>", None, None

    res = ANALYZER.analyze_text(text)

    if "error" in res:
        return f"<div class='disclaimer-box'>Error: {res['error']}</div>", None, None

    # Determine core color
    sev = res['core_severity']['level']
    colors = {"Severe Crisis": "#D32F2F", "Moderate": "#FF9800", "Mild": "#FFC107", "No Concern": "#4CAF50"}
    core_color = colors.get(sev, "#888")

    # 1. Summary HTML
    html = f"""
    <div style="border-top: 5px solid {core_color}; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h2 style="margin-top:0;">Overall Assessment: <span style="color: {core_color};">{sev}</span></h2>
        <p>Confidence: <b>{res['core_severity']['confidence']:.0%}</b> &nbsp;|&nbsp; Source: <b>{res['core_severity']['source']}</b></p>
    """

    # Display Action Flags if any
    if res['flags']:
        html += "<h3>Action Required:</h3>"
        for flag in res['flags']:
            css_class = "flag-box-critical" if flag['type'] == "critical" else "flag-box-warning"
            html += f"<div class='{css_class}'><strong>{flag['type'].upper()}:</strong> {flag['message']}</div>"
    else:
        html += "<p style='color: #4CAF50; font-weight:600;'>No immediate action flags generated.</p>"

    html += "</div>"

    # 2. Charts
    dims = res['dimensions']
    fig_bar = None
    fig_radar = None

    if PLOTLY_AVAILABLE and dims:
        labels = [d['label'] for k, d in dims.items()]
        scores = [d['score'] for k, d in dims.items()]
        dim_colors = [d['color'] for k, d in dims.items()]

        fig_bar = px.bar(
            x=labels, y=scores,
            color=labels,
            color_discrete_sequence=dim_colors,
            labels={'x': 'Emotional Dimension', 'y': 'Intensity (0-100)'},
            title="Emotional Intensity Profile"
        )
        fig_bar.update_layout(yaxis_range=[0, 100], showlegend=False,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=labels + [labels[0]],
            fill='toself',
            line_color=core_color
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, title="Dimensional Balance",
            paper_bgcolor='rgba(0,0,0,0)'
        )
    else:
        # Fallback to HTML list
        html += "<h3>Dimensional Scores:</h3><ul>"
        for k, d in dims.items():
            html += f"<li><b>{d['label']}</b>: {d['score']}/100"
            if d['triggers_found']:
                html += f" <i>(Detected: {', '.join(d['triggers_found'])})</i>"
            html += "</li>"
        html += "</ul>"

    return html, fig_bar, fig_radar


# ============================================================
# Core Logic: Batch Analysis
# ============================================================

def process_batch_texts(text_blob: str):
    """Process multiple texts pasted together, calculate aggregates."""
    if not ANALYZER:
        return "Analyzer not loaded.", None, None, ""

    if not text_blob or len(text_blob.strip()) < 10:
        return "Please paste some anonymous check-in texts (separated by blank lines or '---').", None, None, ""

    # Split by separator "---" or by double newlines
    separator = CFG.get('batch', {}).get('separator', '---')
    if separator in text_blob:
        texts = [t.strip() for t in text_blob.split(separator) if t.strip()]
    else:
        texts = [t.strip() for t in text_blob.split('\n\n') if t.strip()]

    if not texts:
        return "No usable text blocks found. Separate inputs with empty lines or '---'.", None, None, ""

    res = ANALYZER.analyze_batch(texts)

    if "error" in res:
        return f"Error: {res['error']}", None, None, ""

    summ = res['summary']

    # 1. Build Top-Level Stats HTML
    health_color = "#4CAF50" if summ['overall_health_score'] > 75 else ("#FF9800" if summ['overall_health_score'] > 50 else "#D32F2F")

    stats_html = f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div class="stat-card" style="flex: 1; min-width: 200px;">
            <div class="stat-label">Total Responses</div>
            <div class="stat-value">{summ['total_analyzed']}</div>
        </div>
        <div class="stat-card" style="flex: 1; min-width: 200px; border-top: 4px solid {health_color};">
            <div class="stat-label">Community Health Score</div>
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
    fig_sev = None
    fig_dim = None

    if PLOTLY_AVAILABLE:
        # Severity Donut Chart
        sev_counts = res['severity_distribution']
        sev_colors = {"Severe Crisis": "#D32F2F", "Moderate": "#FF9800", "Mild": "#FFC107", "No Concern": "#4CAF50"}
        labels_sev = list(sev_counts.keys())
        values_sev = list(sev_counts.values())
        marker_colors = [sev_colors.get(l, '#888') for l in labels_sev]

        fig_sev = go.Figure(data=[go.Pie(labels=labels_sev, values=values_sev, hole=.4, marker_colors=marker_colors)])
        fig_sev.update_layout(title="Overall Severity Distribution", paper_bgcolor='rgba(0,0,0,0)')

        # Dimensions Bar Chart (Averages)
        avg_dims = res['average_emotional_dimensions']
        dim_labels = [ANALYZER.dimensions.get(k, {}).get('label', k) for k in avg_dims.keys()]
        dim_values = list(avg_dims.values())
        dim_colors = [ANALYZER.dimensions.get(k, {}).get('color', '#888') for k in avg_dims.keys()]

        fig_dim = px.bar(
            x=dim_labels, y=dim_values,
            color=dim_labels,
            color_discrete_sequence=dim_colors,
            labels={'x': 'Emotional Dimension', 'y': 'Average Intensity (0-100)'},
            title="Community Average Emotional Profile"
        )
        fig_dim.update_layout(yaxis_range=[0, 100], showlegend=False,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # Build the textual summary table
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

        # HEADER
        gr.Markdown(f"""
        # {COMMUNITY_NAME}
        ### Aggregated Anonymous Emotional Wellness Insights for Organizations
        """)

        gr.Markdown(f"**Disclaimer:** {DISCLAIMER}", elem_classes=["disclaimer-box"])

        # TABS
        with gr.Tabs():

            # --- TAB 1: BATCH DASHBOARD (Primary Use Case) ---
            with gr.Tab("Community Batch Dashboard"):
                gr.Markdown("Analyze multiple anonymous check-ins simultaneously to generate **aggregate, privacy-preserving** insights. No individual texts are stored or returned.")

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input = gr.Textbox(
                            label="Paste Anonymous Check-ins",
                            placeholder="Paste texts here.\nSeparate distinct responses with an empty line or '---'.\n\nExample:\nI feel stressed and anxious about finals.\n---\nHad a great day today, feeling happy!\n---\nEverything feels pointless and I am so lonely.",
                            lines=12
                        )
                        run_batch_btn = gr.Button("Generate Aggregate Insights", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        batch_stats_html = gr.HTML("<p style='color:#666; padding:20px;'>Submit anonymous responses to see aggregate health statistics.</p>")

                with gr.Row():
                    batch_plot_sev = gr.Plot(label="Severity Distribution")
                    batch_plot_dim = gr.Plot(label="Average Emotional Profile")

                batch_table_md = gr.Markdown("")

                # Event binding
                run_batch_btn.click(
                    fn=process_batch_texts,
                    inputs=[batch_input],
                    outputs=[batch_stats_html, batch_plot_sev, batch_plot_dim, batch_table_md]
                )

            # --- TAB 2: SINGLE TEXT INVESTIGATOR ---
            with gr.Tab("Single Text Investigator"):
                gr.Markdown("Deep dive into a specific anonymous response. Generates detailed multi-dimensional emotional scoring across 8 dimensions.")

                with gr.Row():
                    with gr.Column(scale=1):
                        single_input = gr.Textbox(
                            label="Individual Anonymous Text",
                            placeholder="Enter a single anonymous text to evaluate all 8 emotional dimensions...",
                            lines=8
                        )
                        with gr.Row():
                            run_single_btn = gr.Button("Analyze Text", variant="primary")
                            clear_single_btn = gr.Button("Clear")

                    with gr.Column(scale=1):
                        single_html = gr.HTML("<p style='color:#666;'>Result will appear here.</p>")

                with gr.Row():
                    single_bar = gr.Plot(label="Intensity Profile")
                    single_radar = gr.Plot(label="Dimensional Balance")

                # Event binding
                run_single_btn.click(
                    fn=process_single_text,
                    inputs=[single_input],
                    outputs=[single_html, single_bar, single_radar]
                )
                clear_single_btn.click(
                    fn=lambda: ("", "<p style='color:#666;'>Cleared.</p>", None, None),
                    inputs=[],
                    outputs=[single_input, single_html, single_bar, single_radar]
                )

            # --- TAB 3: REST API INTEGRATION ---
            with gr.Tab("System API"):
                gr.Markdown("""
                # REST API Integration
                Configure your existing anonymous feedback forms (Google Forms, SurveyMonkey, Typeform, etc.)
                to send data to the local MindGuard Community API for automated batch processing.

                ### Endpoints
                - `POST /api/v1/analyze/single` - Real-time single-text screening
                - `POST /api/v1/analyze/batch` - Aggregate privacy-preserving batch analysis (preferred)

                ### Authentication
                Requires header: `X-API-Key: demo-organization-key`

                ### Example cURL
                ```
                curl -X POST "http://localhost:8000/api/v1/analyze/batch" \\
                     -H "Content-Type: application/json" \\
                     -H "X-API-Key: demo-organization-key" \\
                     -d '{"texts": ["I feel stressed", "Having a great day"]}'
                ```
                """)

                api_log = gr.Textbox(label="Server Status", value="Stopped", interactive=False)
                start_api_btn = gr.Button("Start Local API Server", variant="primary")

                start_api_btn.click(
                    fn=start_api_server,
                    inputs=[],
                    outputs=[api_log]
                )

        # FOOTER
        gr.Markdown(f"""
        ---
        **Privacy Notice:** No individual text is stored or logged. All analysis is performed in-memory and only aggregate statistics are displayed.
        This is a **screening tool only** and does NOT diagnose any mental health condition. All flags must be reviewed by a qualified professional.

        *{COMMUNITY_NAME} v2.0 | Privacy-Preserving Emotional Wellness Platform*
        """)

    return app


if __name__ == "__main__":
    print("[INFO] Starting MindGuard Community Dashboard...")
    app = create_dashboard()
    if app:
        app.launch(server_port=7865, share=False, show_error=True)

import sys
import threading
from pathlib import Path

def start_api():
    port = analyzer.config.get('api', {}).get('port', 8000)
    print(f"\n🚀 Starting MindGuard Community API on port {port}...")
    
    # Run uvicorn programmatically
    import uvicorn
    uvicorn.run("api.community_api:app", host="127.0.0.1", port=port, log_level="warning")

def create_app():
    # ... inside your Gradio block:
    # Add an API Settings/Info tab
    
    with gr.Tab("🔌 API Integration"):
        gr.Markdown("""
        # Organization REST API
        MindGuard Community Edition provides a fully private REST API so you can connect your existing survey tools (Google Forms, Qualtrics, internal portals) directly to our screening engine.
        
        ## Endpoints
        - `POST /api/v1/analyze/single`: Real-time screening
        - `POST /api/v1/analyze/batch`: Aggregate privacy-preserving insights
        
        *Your data is processed in-memory and never logged.*
        """)
        
        api_status = gr.Textbox(label="API Status", value="Stopped", interactive=False)
        start_btn = gr.Button("▶️ Start Local API Server")
        
        def toggle_api():
            thread = threading.Thread(target=start_api, daemon=True)
            thread.start()
            return "✅ Running on http://127.0.0.1:8000"
            
        start_btn.click(fn=toggle_api, outputs=[api_status])

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist, Field
from typing import List, Dict, Optional
import os
import secrets
from model.emotional_analyzer import EmotionalAnalyzer
import time

# Load configuration
analyzer = EmotionalAnalyzer()
config = analyzer.config.get('api', {})

app = FastAPI(
    title=analyzer.config.get('community', {}).get('name', 'MindGuard Community API'),
    description=analyzer.config.get('community', {}).get('tagline', 'Anonymous emotional wellness insights'),
    version=analyzer.config.get('community', {}).get('version', '2.0.0'),
)

# CORS Setup
origins = config.get('cors_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# API Key Authentication (Basic implementation)
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = {os.getenv("MINDGUARD_API_KEY", "demo-organization-key")} # Load from env in production

def verify_api_key(request: Request):
    if not config.get('require_api_key', True):
        return True
        
    api_key = request.headers.get(API_KEY_HEADER)
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return True

# Data Models
class BatchAnalysisRequest(BaseModel):
    # Enforce batch limits to prevent abuse
    texts: conlist(str, min_items=1, max_items=analyzer.config.get('batch', {}).get('max_texts_per_batch', 100))
    organization_id: Optional[str] = Field(None, description="Optional org tracking for analytics")

class SingleAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)

@app.get("/")
async def root():
    return {
        "status": "online", 
        "service": app.title,
        "disclaimer": analyzer.config.get('community', {}).get('ethical_disclaimer', '')
    }

@app.post("/api/v1/analyze/single")
async def analyze_single(request: SingleAnalysisRequest, _ = Depends(verify_api_key)):
    """
    Analyze a single anonymous text. 
    Use for real-time form submission checking.
    """
    start_time = time.time()
    result = analyzer.analyze_text(request.text)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return {
        "analysis": result,
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "disclaimer": "This is a screening score only. Not diagnostic."
    }

@app.post("/api/v1/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest, _ = Depends(verify_api_key)):
    """
    Analyze a batch of texts from a community over a period (e.g., weekly pulse surveys).
    Returns ONLY aggregate statistics due to strict privacy constraints.
    """
    start_time = time.time()
    aggregate_results = analyzer.analyze_batch(request.texts)
    
    if "error" in aggregate_results:
        raise HTTPException(status_code=400, detail=aggregate_results["error"])
        
    return {
        "organization_id": request.organization_id or "anonymous",
        "timestamp": int(time.time()),
        "aggregate_insights": aggregate_results,
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "privacy_notice": "Individual texts and results are not stored or returned."
    }

if __name__ == "__main__":
    import uvicorn
    # Make sure we use the configured port
    port = config.get('port', 8000)
    print(f"🚀 Starting MindGuard Community API on port {port}...")
    uvicorn.run("api.community_api:app", host="0.0.0.0", port=port, reload=True)

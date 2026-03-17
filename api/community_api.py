from fastapi import FastAPI, HTTPException, Request, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist, Field
from typing import List, Dict, Optional, Any
import os
import time

# Load config and core modules
from model.emotional_analyzer import EmotionalAnalyzer
from db.database import MindGuardDB

# Initialize services
analyzer = EmotionalAnalyzer()
config = analyzer.config
db = MindGuardDB(config.get('database', {}).get('path', 'data/mindguard.db'))

app = FastAPI(
    title=config.get('community', {}).get('name', 'MindGuard Community API'),
    description=config.get('community', {}).get('tagline', 'Anonymous emotional wellness insights'),
    version=config.get('community', {}).get('version', '3.0.0'),
)

# CORS Setup
origins = config.get('api', {}).get('cors_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

API_KEY_HEADER = "X-API-Key"

# ==========================================================
# Authentication & Authorization
# ==========================================================

async def verify_api_key(request: Request) -> Dict:
    """Validate API key against database and check rate limits."""
    if not config.get('api', {}).get('require_api_key', True):
        # Fallback for local testing if auth disabled
        return {"org_id": "local_dev", "tier": "free", "org_name": "Local Dev"}
        
    api_key = request.headers.get(API_KEY_HEADER)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide 'X-API-Key' header.",
        )
        
    # Check DB
    key_info = db.validate_api_key(api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API Key.",
        )
        
    org_id = key_info['org_id']
    tier = key_info['tier']
    
    # Check rate limit/pricing tier limits
    tier_limits = config.get('pricing', {}).get('tiers', {}).get(tier, {})
    limit_info = db.check_rate_limit(org_id, tier_limits)
    
    if not limit_info['allowed']:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly analysis limit reached for tier '{tier}'. Upgrade at mindguard.ai/billing.",
        )
        
    return key_info

# ==========================================================
# Data Models
# ==========================================================

class SingleAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)

class BatchAnalysisRequest(BaseModel):
    # Dynamic batch limit max_items not supported directly in Pydantic v2 via instance variable,
    # so we enforce it in the handler
    texts: List[str] = Field(..., min_length=1)
    save_snapshot: bool = Field(True, description="Save aggregate trend to dashboard history")

class WebhookPayload(BaseModel):
    event_type: str
    org_id: str
    timestamp: int
    data: Dict[str, Any]

# ==========================================================
# Background Tasks
# ==========================================================

def trigger_webhooks(org_id: str, event_type: str, data: Dict):
    """Fire registered webhooks asynchronously."""
    if not config.get('webhooks', {}).get('enabled', False):
        return
        
    import httpx
    webhooks = db.get_webhooks(org_id, event_type)
    
    if not webhooks:
        return
        
    payload = WebhookPayload(
        event_type=event_type,
        org_id=org_id,
        timestamp=int(time.time()),
        data=data
    ).model_dump()
    
    for wh in webhooks:
        # Simplistic webhook retry & push
        try:
            with httpx.Client(timeout=10) as client:
                client.post(wh['url'], json=payload)
        except Exception as e:
            # In a real app, queue this for retry via Celery/Redis
            print(f"Webhook delivery failed to {wh['url']}: {e}")

# ==========================================================
# Endpoints
# ==========================================================

@app.get("/")
async def root():
    return {
        "status": "online", 
        "service": app.title,
        "version": app.version,
        "disclaimer": config.get('community', {}).get('ethical_disclaimer', '')
    }

@app.post("/api/v1/analyze/single")
async def analyze_single(
    request: SingleAnalysisRequest, 
    background_tasks: BackgroundTasks,
    org_info: Dict = Depends(verify_api_key)
):
    """
    Analyze a single anonymous text in real-time.
    (Included in all plans)
    """
    start_time = time.time()
    org_id = org_info['org_id']
    
    # Process
    result = analyzer.analyze_text(request.text)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    # Log usage
    background_tasks.add_task(db.log_usage, org_id=org_id, endpoint="/analyze/single", count=1)
    
    return {
        "organization_id": org_id,
        "analysis": result,
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "disclaimer": "This is a screening score only. Not diagnostic."
    }

@app.post("/api/v1/analyze/batch")
async def analyze_batch(
    request: BatchAnalysisRequest, 
    background_tasks: BackgroundTasks,
    org_info: Dict = Depends(verify_api_key)
):
    """
    Analyze a batch of anonymous texts and return AGGREGATE statistics.
    Individual texts and predictions are purposefully dropped to preserve privacy.
    (Included in all plans, limits vary by tier)
    """
    start_time = time.time()
    org_id = org_info['org_id']
    tier = org_info['tier']
    
    # Enforce batch size limits by tier
    tier_limits = config.get('pricing', {}).get('tiers', {}).get(tier, {})
    max_batch = tier_limits.get('max_batch_size', 25)
    
    if len(request.texts) > max_batch:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(request.texts)} exceeds your tier limit ({max_batch})."
        )
        
    aggregate_results = analyzer.analyze_batch(request.texts)
    
    if "error" in aggregate_results:
        raise HTTPException(status_code=400, detail=aggregate_results["error"])
        
    # Log usage
    background_tasks.add_task(db.log_usage, org_id=org_id, endpoint="/analyze/batch", count=len(request.texts))
    
    summary = aggregate_results['summary']
    
    # Save trend snapshot for dashboard (if requested)
    if request.save_snapshot:
        background_tasks.add_task(
            db.save_trend_snapshot,
            org_id=org_id,
            total_analyzed=summary['total_analyzed'],
            health_score=summary['overall_health_score'],
            critical_flags=summary['critical_flags_generated'],
            severity_distribution=aggregate_results['severity_distribution'],
            avg_dimensions=aggregate_results['average_emotional_dimensions']
        )
        
    # Webhooks
    background_tasks.add_task(
        trigger_webhooks, 
        org_id=org_id, 
        event_type="batch_analysis_complete", 
        data=aggregate_results
    )
    
    return {
        "organization_id": org_id,
        "organization_name": org_info['org_name'],
        "tier": tier,
        "timestamp": int(time.time()),
        "aggregate_insights": aggregate_results,
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "privacy_notice": "Individual texts and results are not stored."
    }

@app.get("/api/v1/trends")
async def get_trends(
    days: int = 30,
    org_info: Dict = Depends(verify_api_key)
):
    """Retrieve aggregate trend history for the organization dashboard."""
    org_id = org_info['org_id']
    tier = org_info['tier']
    
    # Check feature access
    if tier == 'free':
        raise HTTPException(status_code=403, detail="Trend history requires Starter tier or higher.")
        
    # Max days by tier
    max_days = 30 if tier == 'starter' else 365
    days = min(days, max_days)
    
    history = db.get_trend_history(org_id, days=days)
    
    return {
        "organization_id": org_id,
        "period_days": days,
        "snapshots": history
    }

if __name__ == "__main__":
    import uvicorn
    port = config.get('api', {}).get('port', 8000)
    print(f"🚀 Starting MindGuard Community API v3 on port {port}...")
    uvicorn.run("api.community_api:app", host="0.0.0.0", port=port, reload=True)

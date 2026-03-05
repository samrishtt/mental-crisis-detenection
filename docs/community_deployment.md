# MindGuard Community Edition — Deployment Guide

## Overview

MindGuard Community Edition transforms the original one-on-one school counselor tool into an organization-wide platform for tracking the emotional wellness of entire communities (schools, universities, workplaces).

### Key Features of Community Edition
1. **Multi-dimensional Emotional Profiling** (8 dimensions: Anxiety, Depression, Grief, Burnout, Loneliness, Anger, Positive Wellness, Core Severity)
2. **Batch Aggregate Analysis Dashboard** 
3. **REST API Integration**
4. **Absolute Privacy Enforcement** (Only macro trends are stored)

---

## 🚀 Quick Start Guide

### 1. Setup Environment
```bash
pip install -r requirements.txt
pip install fastapi uvicorn pydantic plotly python-multipart
```

### 2. Configure Your Organization
Edit `config/community_config.yaml` to specify your organization's name, emotional trigger keywords, and API settings.

```yaml
community:
  name: "Springfield High School Wellness"
```

### 3. Launch Dashboard & API
Start the unified Web UI. This interface allows you to view the dashboard and click "Start Local API Server" to run the REST endpoint for surveying.

```bash
python app/community_dashboard.py
```

*The interface will run on `http://localhost:7860`*

---

## 🔒 Ethical and Privacy Considerations

Because mental health data is extremely sensitive, the Community Edition enforces strict constraints:

1. **No Data Storage At Rest**: The `EmotionalAnalyzer` processes texts entirely in memory.
2. **The Batch API Constraint**: The `/api/v1/analyze/batch` endpoint takes a list of anonymous inputs but **removes individual results** before sending the response back. It ONLY returns aggregate means and counts.
3. **Action Triggers over Identifiers**: The system will flag if a "Critical" keyword pattern (e.g., self-harm) occurs "N times", but it deliberately does NOT tell the admin *which* text triggered it, preventing them from reverse-identifying anonymous respondents.

> **Why?** Identifying an individual behind an anonymous submission violates trust. If a severe crisis is detected in aggregate form, standard procedure is to broadcast universal support resources to the whole group, rather than attempting to track down an individual.

---

## 🔌 Integrating the REST API

If you use tools like Google Forms, SurveyMonkey, or custom apps for anonymous check-ins, you can configure them to ping the `/analyze/batch` endpoint.

**Webhook Example Configuration (cURL):**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/batch" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: demo-organization-key" \
     -d '{
           "organization_id": "org_1234",
           "texts": [
             "I feel really burned out and exhausted lately.",
             "Doing great, had a fun weekend!",
             "Just anxious about the math test tomorrow."
           ]
         }'
```

The system will ingest the array and return:
```json
{
  "aggregate_insights": {
    "summary": {
      "total_analyzed": 3,
      "overall_health_score": 85
    },
    "severity_distribution": {
      "No Concern": 1,
      "Mild": 2,
      "Moderate": 0,
       "Severe Crisis": 0
    },
    "average_emotional_dimensions": {
      "anxiety": 6.6,
      "burnout": 6.6,
      "positive_wellness": 13.3
    }
  }
}
```

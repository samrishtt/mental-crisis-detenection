FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Set work directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to cache them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn httpx gradio plotly

# Apply application code
COPY . .

# Create necessary directories
RUN mkdir -p data checkpoints results logs \
    && chmod -R 777 data checkpoints results logs

# Expose ports:
# 8000: Community REST API
# 7861: Admin Dashboard (Internal Only)
# 7862: Community Dashboard
EXPOSE 8000 7861 7862

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting MindGuard Commercial Stack..."\n\
uvicorn api.community_api:app --host 0.0.0.0 --port 8000 &\n\
python app/admin_dashboard.py &\n\
python app/community_dashboard.py &\n\
wait\n\
' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

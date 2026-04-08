# Use a slim Python 3.11 image
FROM python:3.11-slim

# Install uv system-wide
RUN pip install uv

# Set up working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using uv into the system environment
RUN uv pip install --system "fastapi>=0.111.0" "uvicorn>=0.30.1" "pydantic>=2.7.4" "httpx>=0.27.0" "python-multipart>=0.0.9" "python-dotenv>=1.0.1"

# Copy application code
COPY ./app ./app

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

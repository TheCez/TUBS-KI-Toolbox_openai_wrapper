# Use a slim Python 3.11 image
FROM python:3.11-slim

# Install uv system-wide
RUN pip install uv

# Set up working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies from pyproject.toml using uv
RUN uv pip install --system .

# Copy application code
COPY ./app ./app

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

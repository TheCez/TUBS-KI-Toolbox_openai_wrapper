# Use a slim Python 3.11 image
FROM python:3.11-slim

# Install uv from official image (faster & smaller than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up working directory
WORKDIR /app

# Copy dependency and project metadata files
COPY pyproject.toml README.md ./

# Install dependencies only (no need to build the project as a wheel)
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY ./app ./app

# Expose port
EXPOSE 8000

# Use uv run to execute within the managed virtual environment
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

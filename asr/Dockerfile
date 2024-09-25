# Dockerfile
FROM python:3.9

WORKDIR /app

# Install necessary libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY asr_api.py .

# Expose the necessary port
EXPOSE 8001

# Run the FastAPI server
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]

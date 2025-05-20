# Get python image (latest version)
FROM python:latest

# Set working directory as the /app
WORKDIR /app

# Copy the contents of the current path to /app
COPY . .

# Install the requirements
RUN pip install -r requirements.txt

# Expose port 8000
EXPOSE 8000


# Run FastAPI app using Uvicorn
CMD ["uvicorn", "apis:app", "--host", "0.0.0.0", "--port", "8000"]
# Use the official Python image as the base image
FROM python:3.9-alpine

RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y gcc

# Copy the entire project into the image
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

# Expose the ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5000

# Command to run MLflow Tracking Server and Streamlit app
CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns --host 0.0.0.0 --port 5000 & streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0

# Use the official Python image as the base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the entire project into the image
COPY . /app

# Copy Nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Install the dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

# Expose the ports for Nginx
EXPOSE 8080

# Command to run Nginx and the app
CMD service nginx start && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns --host 0.0.0.0 --port 5000 & streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0

# MLOps Project

This project demonstrates a simple MLOps pipeline for training, validating, and deploying a machine learning model using PyTorch and Flask.

## Project Structure

- `data/`: Contains the dataset file.
- `models/`: Contains the saved model.
- `src/`: Contains the source code for training, model definition, and deployment.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis.
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Project documentation.

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train.py
   ```

3. Run the Flask app:
   ```bash
   python src/app.py
   ```

4. Make a prediction:
   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
   ```

import pandas as pd
import torch
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_test_data():
    # Load test data
    test_X = pd.read_csv('../data/processed/test_X.csv').values
    test_y = pd.read_csv('../data/processed/test_y.csv').values.ravel()
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.long)
    return test_X, test_y

def evaluate_model():
    # Load model dari MLflow menggunakan tag
    client = mlflow.tracking.MlflowClient()
    model_info = client.search_model_versions(f"name='IrisModel' and tags.stage='Production'")[0]
    model_uri = f"models:/{model_info.name}/{model_info.version}"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Memuat data uji
    test_X, test_y = load_test_data()
    
    # Evaluasi model
    model.eval()
    with torch.no_grad():
        outputs = model(test_X)
        _, preds = torch.max(outputs, 1)
    
    # Hitung metrik kinerja
    accuracy = accuracy_score(test_y, preds)
    precision = precision_score(test_y, preds, average='macro')
    recall = recall_score(test_y, preds, average='macro')
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    evaluate_model()

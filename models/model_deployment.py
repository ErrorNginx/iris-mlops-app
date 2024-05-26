import mlflow
import mlflow.pytorch

def deploy_model():
    # Load model dari MLflow menggunakan tag
    client = mlflow.tracking.MlflowClient()
    model_info = client.search_model_versions(f"name='IrisModel' and tags.stage='Production'")[0]
    model_uri = f"models:/{model_info.name}/{model_info.version}"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Save the model to a file for deployment
    mlflow.pytorch.save_model(model, "deployed_model")

if __name__ == "__main__":
    deploy_model()

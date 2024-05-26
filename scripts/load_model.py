import mlflow.pytorch

def load_model(model_uri):
    model = mlflow.pytorch.load_model(model_uri)
    return model

if __name__ == "__main__":
    model_uri = "models:/IrisModel/Production"
    model = load_model(model_uri)
    print("Model loaded successfully.")

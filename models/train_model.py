import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)
        return X

# Load and preprocess the data
def load_data():
    dataset = pd.read_csv('data/raw/iris.csv')
    dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2
    
    # Ensure the species column is of integer type
    dataset['species'] = dataset['species'].astype(int)

    train_X, test_X, train_y, test_y = train_test_split(
        dataset[dataset.columns[0:4]].values,
        dataset.species.values,
        test_size=0.2
    )
    
    # Convert to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_y = torch.tensor(test_y, dtype=torch.long)
    
    return train_X, test_X, train_y, test_y

# Training the model
def train_model():
    # Create or set an MLflow experiment
    experiment_name = "Iris_Experiment"
    mlflow.set_experiment(experiment_name)

    train_X, test_X, train_y, test_y = load_data()
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    with mlflow.start_run() as run:
        for epoch in range(1000):  # Perbaiki tanda kurung
            net.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if epoch % 100 == 0:
                net.eval()
                with torch.no_grad():
                    test_preds = []
                    test_targets = []
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = net(inputs)
                        _, predicted = torch.max(outputs, 1)
                        test_preds.extend(predicted.cpu().numpy())
                        test_targets.extend(labels.cpu().numpy())
                accuracy = accuracy_score(test_targets, test_preds)
                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_metric('loss', loss.item())
                print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}')

        net.eval()
        with torch.no_grad():
            test_preds = []
            test_targets = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        accuracy = accuracy_score(test_targets, test_preds)
        macro_precision = precision_score(test_targets, test_preds, average='macro')
        micro_precision = precision_score(test_targets, test_preds, average='micro')
        macro_recall = recall_score(test_targets, test_preds, average='macro')
        micro_recall = recall_score(test_targets, test_preds, average='micro')

        print(f'Prediction accuracy: {accuracy}')
        print(f'Macro precision: {macro_precision}')
        print(f'Micro precision: {micro_precision}')
        print(f'Macro recall: {macro_recall}')
        print(f'Micro recall: {micro_recall}')

        mlflow.log_metric('final_accuracy', accuracy)
        mlflow.log_metric('macro_precision', macro_precision)
        mlflow.log_metric('micro_precision', micro_precision)
        mlflow.log_metric('macro_recall', macro_recall)
        mlflow.log_metric('micro_recall', micro_recall)
        
        # Log the model
        mlflow.pytorch.log_model(net, "model")
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name="IrisModel")

        # Add tags to indicate stage
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(name="IrisModel", version=registered_model.version, key="stage", value="Production")

if __name__ == "__main__":
    train_model()

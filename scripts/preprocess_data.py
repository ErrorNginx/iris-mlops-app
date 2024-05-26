import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Memuat data mentah
    data = pd.read_csv('./data/raw/iris.csv')
    
    # Transformasi variabel target
    data['species'] = data['species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    
    # Split data menjadi training dan testing
    X = data.drop(columns='species')
    y = data['species']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    print(test_y)

    # Buat direktori jika belum ada
    os.makedirs('../data/processed', exist_ok=True)
    
    # Simpan data preprocessed
    train_X.to_csv('../data/processed/train_X.csv', index=False)
    test_X.to_csv('../data/processed/test_X.csv', index=False)
    train_y.to_csv('../data/processed/train_y.csv', index=False)
    test_y.to_csv('../data/processed/test_y.csv', index=False)

if __name__ == "__main__":
    preprocess_data()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def model_train(data_path, model_path):
    df = pd.read_csv(data_path)

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33, stratify=y)

    model = RandomForestClassifier(criterion='gini',
                                   max_depth=20,
                                   max_features='log2',
                                   min_samples_leaf=2,
                                   min_samples_split=20)

    model.fit(X_train, y_train)

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    print(f"Model trained and saved to {model_path}")

def main():
    data_path = "artifacts/Optimizing Agriculture Production.csv"
    model_path = "artifacts/model.pkl"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_train(data_path, model_path)

if __name__ == "__main__":
    main()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from xgboost import XGBClassifier

import numpy as np

from utils import *

if __name__ == "__main__":
    dataset = "FAVA"

    model_name = "your_model_name"      # Replace this the name of the model you want to run with
    cache_dir = "your_cache_dir"        # Replace this with your cache directory path
    
    if dataset == "FAVA":
        data = read_fava_data(n_samples=200)
        tokenizer, model = load_model(model_name, cache_dir)
        h0_bars, h1_bars, labels = get_bars_from_data(data, model_name, cache_dir, tokenizer)

        pers_vectors = vectorize_persistence_diagram(h1_bars, method="persistence_image")
        
        X_train, X_test, y_train, y_test = train_test_split(
            pers_vectors, labels, test_size=0.2, random_state=42
        )
    elif dataset == "RAGTruth":
        train_data, test_data = get_ragtruth_data(n_samples=200)
        tokenizer, model = load_model(model_name, cache_dir)
        h0_bars_train, h1_bars_train, labels_train = get_bars_from_data(train_data, model_name, cache_dir, tokenizer)

        h0_bars_test, h1_bars_test, labels_test = get_bars_from_data(test_data, model_name, cache_dir, tokenizer)

        pers_vectors_train = vectorize_persistence_diagram(h1_bars_train, method="persistence_image")
        pers_vectors_test = vectorize_persistence_diagram(h1_bars_test, method="persistence_image")

        X_train, X_test, y_train, y_test = pers_vectors_train, pers_vectors_test, labels_train, labels_test

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
    print("F1 Score:", f1_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    tpr_at_5_fpr = 0.0
    for f, t in zip(fpr, tpr):
        if f >= 0.05:
            tpr_at_5_fpr = t
            break
    print(f"TPR at 5% FPR:", {tpr_at_5_fpr})
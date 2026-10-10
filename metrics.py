# src/metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model_performance(y_true, y_pred):
    """
    Calcula y retorna un diccionario con las métricas estándar de evaluación:
    Accuracy, Precision, Recall y F1-Score (ponderados).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    }
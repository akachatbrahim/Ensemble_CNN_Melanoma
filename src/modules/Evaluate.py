import numpy as np
from src.config.Config import learning_rate, weight_decay,epochs,threshold,metrics_per_model,ml_models
from sklearn.metrics import roc_auc_score,classification_report,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def print_classification_report_percentage(y_true, y_pred, target_names=["benign", "malignant"]):
    """Print classification report with percentages (2 decimal places)"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Calculate macro average (unweighted)
    macro_precision = np.mean(precision) * 100
    macro_recall = np.mean(recall) * 100
    macro_f1 = np.mean(f1) * 100
    
    # Calculate weighted average
    weighted_precision = np.average(precision, weights=support) * 100
    weighted_recall = np.average(recall, weights=support) * 100
    weighted_f1 = np.average(f1, weights=support) * 100
    
    print("\n" + "="*70)
    print(f"{'':20} {'Precision':>15} {'Recall':>15} {'F1-Score':>15} {'Support':>10}")
    print("="*70)
    
    for i, target_name in enumerate(target_names):
        print(f"{target_name:20} {precision[i]*100:14.2f}% {recall[i]*100:14.2f}% {f1[i]*100:14.2f}% {int(support[i]):10}")
    
    print("-"*70)
    print(f"{'accuracy':20} {accuracy:>15.2f}%")
    print(f"{'macro avg':20} {macro_precision:14.2f}% {macro_recall:14.2f}% {macro_f1:14.2f}% {int(support.sum()):10}")
    print(f"{'weighted avg':20} {weighted_precision:14.2f}% {weighted_recall:14.2f}% {weighted_f1:14.2f}% {int(support.sum()):10}")
    print("="*70)

def soft_voting(models, dataset, threshold=threshold):
    y_true = []
    y_probs = []

    for x, y in dataset:
        probs = [model.predict(x, verbose=0) for model in models]
        avg_prob = np.mean(probs, axis=0)

        y_probs.extend(avg_prob.ravel())
        y_true.extend(y.numpy().ravel())

    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    return np.array(y_true), y_pred, y_probs
    
def get_predictions(model, dataset):
    y_true, y_pred = [], []

    for x, y in dataset:
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy().ravel())
        y_pred.extend((preds > threshold).astype(int).ravel())

    return np.array(y_true).astype(int), np.array(y_pred)

def hybrid_model(modelname):
    logique_return = False
    if "ML" in modelname:
        logique_return = True
    return logique_return 

def ensemble_model(modelname):
    logique_return = False
    if "ENSEMBLE" in modelname:
        logique_return = True
    return logique_return    

# Function to evaluate model
def evaluate_model(model,model_name,X_data,y_data,data,ml_models):
    print(f"Model:{model_name} | ", f"epochs:{epochs} | ",     f"learning_rate:{learning_rate:.1e} | ",     f"weight_decay:{weight_decay:.1e}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    if hybrid_model(model_name):
        y_probs = model.predict_proba(X_data)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        y_true = np.array(y_data).ravel()  # Ensure y_data is flattened to 1D
        y_scores = y_probs
    else: 
        if ensemble_model(model_name):
            y_true, y_pred, y_probs = soft_voting(model, data, threshold)
            y_scores = y_probs  # Use y_probs from soft voting for ensemble models
        else:     
            y_true, y_pred = get_predictions(model, data)
            y_scores = model.predict(data).ravel()  # Probabilité pour la classe "malignant" (1)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy    = round((tp + tn) / (tp + tn + fp + fn) * 100, 2)
    sensitivity = round(tp / (tp + fn) * 100, 2)          # Recall (Malignant)
    specificity = round(tn / (tn + fp) * 100, 2)          # Recall (Benign)
    #accuracy = accuracy_score(y_true, y_pred)
    precision   = round(tp / (tp + fp) * 100, 2)
    f1          = round(2 * precision * sensitivity / (precision + sensitivity), 2)
    auc = round(roc_auc_score(y_true, y_scores) * 100, 2)
    
    # Create metrics dictionary
    metric_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc-roc": auc,
        "f1-score": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }
    
    if hybrid_model(model_name):
        metrics_per_model[model_name] = metric_dict
        # Store trained model
        ml_models[model_name] = model
    else:    
        metrics_per_model[model_name] = metric_dict
        # For ensemble models (list of models), store the model list
        if isinstance(model, list):
            ml_models[model_name] = model
        else:
            ml_models[model_name] = model
    
    print(f"Accuracy:                 {accuracy:.2f}%")
    print(f"Sensitivity (Malignant):  {sensitivity:.2f}%")
    print(f"Specificity (Benign):     {specificity:.2f}%")
    print(f"Precision (Malignant):    {precision:.2f}%")
    print(f"F1-score (Malignant):     {f1:.2f}%")
    print(f"auc-roc: {auc:.2f}%")
    print(f"=== Classification Report: {model_name} ===")
    print_classification_report_percentage(y_true, y_pred, target_names=["benign", "malignant"])
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return ml_models, metrics_per_model
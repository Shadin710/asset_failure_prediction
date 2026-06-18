import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from src.feature_engineering import engineer_features
from src.models import load_model

def generate_results_matrices():
    print("=" * 60)
    print("     PREDICTIVE MAINTENANCE EVALUATION MATRIX ENGINE     ")
    print("=" * 60)

    # 1. Load and Prepare Test Data
    df = pd.read_csv("data/predictive_maintenance.csv")
    df_processed = engineer_features(df)
    
    X = df_processed.drop(columns=['Target', 'Failure Type'])
    y_binary = df_processed['Target']
    y_multi = df_processed['Failure Type']
    
    # FIX: Pass both y_binary and y_multi into ONE split call to align indices perfectly
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multi, test_size=0.2, stratify=y_binary, random_state=42
    )

    # 2. Evaluate Tier 1: Watchdog Binary Model
    print("\n[EVALUATING TIER 1: BINARY WATCHDOG MODEL]")
    try:
        binary_model = load_model("models/tier1_binary_model.pkl")
        y_pred_bin = binary_model.predict(X_test)
        y_probs_bin = binary_model.predict_proba(X_test)[:, 1]
        
        # Calculate Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin, y_probs_bin)
        pr_auc = auc(recall_vals, precision_vals)
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test_bin, y_pred_bin)
        print(f"True Negatives (Predicted Safe, Actually Safe): {cm[0][0]}")
        print(f"False Positives (False Alarms):              {cm[0][1]}")
        print(f"False Negatives (Missed Failures!):          {cm[1][0]} <-- Crucial Metric")
        print(f"True Positives (Predicted Failure, Caught):   {cm[1][1]}")
        
        print("\n--- Classification Metrics Summary ---")
        print(classification_report(y_test_bin, y_pred_bin, target_names=["Operational", "Failure Risk"]))
        print(f"Precision-Recall AUC Score: {pr_auc:.4f}")
        
    except FileNotFoundError:
        print("Error: Tier 1 model artifact missing from /models directory. Run main.py first.")

    # 3. Evaluate Tier 2: Diagnostic Multi-Class Model
    print("\n" + "="*60)
    print("[EVALUATING TIER 2: MULTI-CLASS DIAGNOSTIC MODEL]")
    try:
        diagnostic_model = load_model("models/tier2_diagnostic_model.pkl")
        
        # Evaluate diagnostic capabilities strictly on actual failure cases within test set
        failure_mask = (y_test_bin == 1)
        if failure_mask.sum() > 0:
            X_test_failures = X_test[failure_mask]
            y_test_failures_type = y_test_multi[failure_mask]  # Works flawlessly now!
            
            y_pred_multi = diagnostic_model.predict(X_test_failures)
            
            print("\n--- Diagnostic Breakdown Matrix ---")
            print(classification_report(y_test_failures_type, y_pred_multi, zero_division=0))
        else:
            print("\nNotice: The random 20% test partition contains no physical failure samples to evaluate.")
            
    except FileNotFoundError:
        print("Error: Tier 2 model artifact missing from /models directory. Run main.py first.")
    print("=" * 60)

if __name__ == "__main__":
    generate_results_matrices()
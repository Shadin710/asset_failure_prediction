import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from src.feature_engineering import engineer_features
from src.models import load_model

def generate_performance_plots():
    print("Generating statistical data visualizer plots...")
    
    # 1. Load Data
    df = pd.read_csv("data/predictive_maintenance.csv")
    df_processed = engineer_features(df)
    
    # Plot A: Feature Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    # Select engineered features + target label for matrix
    corr_features = ['Thermal_Diff', 'Mechanical_Power', 'Frictional_Wear_Index', 'Type_Quality', 'Target']
    corr_matrix = df_processed[corr_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Engineered Physical Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("./images/feature_correlation_heatmap.png")
    print("-> Saved: feature_correlation_heatmap.png")
    plt.close()

    # Plot B: Precision-Recall Curve (The Gold Standard for Imbalanced Data)
    X = df_processed.drop(columns=['Target', 'Failure Type'])
    y_binary = df_processed['Target']
    
    _, X_test, _, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
    )
    
    try:
        binary_model = load_model("models/tier1_binary_model.pkl")
        y_probs = binary_model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test_bin, y_probs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
        plt.xlabel("Recall (Sensitivity to catch failures)")
        plt.ylabel("Precision (Accuracy of sounding alarms)")
        plt.title("Tier 1 Binary Model Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("./images/precision_recall_curve.png")
        print("-> Saved: precision_recall_curve.png")
        plt.close()
        
    except FileNotFoundError:
        print("Could not generate PR Curve. Please train models by running main.py first.")

if __name__ == "__main__":
    generate_performance_plots()
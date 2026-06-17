from src.feature_engineering import load_data, engineer_features
from src.models import train_binary_model, train_diagnostic_model, save_model

def run_pipeline():
    print("1. Loading raw data...")
    df = load_data("data/predictive_maintenance.csv")
    
    print("2. Engineering physical features...")
    df_processed = engineer_features(df)
    
    # Define Targets
    X = df_processed.drop(columns=['Target', 'Failure Type'])
    y_binary = df_processed['Target']
    y_multi = df_processed['Failure Type']
    
    print("3. Training Tier 1 Binary Model...")
    binary_model = train_binary_model(X, y_binary)
    save_model(binary_model, "tier1_binary_model.pkl")
    
    print("4. Training Tier 2 Diagnostic Model...")
    # Train only on rows that actually failed
    failure_mask = (y_binary == 1)
    X_failures = X[failure_mask]
    y_failures = y_multi[failure_mask]
    
    diagnostic_model = train_diagnostic_model(X_failures, y_failures)
    save_model(diagnostic_model, "tier2_diagnostic_model.pkl")
    
    print("Pipeline Complete! Models saved to /models.")

if __name__ == "__main__":
    run_pipeline()
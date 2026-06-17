import pickle
from sklearn.ensemble import RandomForestClassifier

def train_binary_model(x_train,y_train):
    model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
    model.fit(x_train,y_train)
    return model

def train_diagnostic_model(x_failure,y_failure):
    # training the diagnostic model why the failure happened
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(x_failure,y_failure)
    return model
def save_model(model,filename):
    with open(f"models/{filename}",'wb') as file:
        pickle.dump(model, file)
def load_model(filepath):
    with open(filepath,'rb') as file:
        return pickle.load(file)


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_classical_svm(X_train, y_train):
    # Classical SVM with RBF Kernel
    
    # Create a pipeline: Standardize features -> SVM
    
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
    clf.fit(X_train, y_train)
    return clf

def evaluate_baseline(model, X_test, y_test):
    # Return Model Accuracy
    
    return model.score(X_test, y_test)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def RPS_count(model, X_test, y_test):
    """
    Counts RPS of the given model and test data

        Parameters:
            model : model with .predict_proba() method
            X_test (DataFrame) : test sample features \n
            y_test (DataFrame) : test sample targets \n

        Returns:
            rps (double) : rps of the provided model
    """
    y_prob = model.predict_proba(X_test)
    y_observ = pd.get_dummies(y_test)
    return np.mean(np.sum((np.cumsum(y_prob, axis=1) - np.cumsum(y_observ, axis=1)) ** 2, axis=1)
                   / (y_observ.shape[1] - 1))


def show_metrics(model, X_test, y_test):
    """
    Prints confusion matrix, classification report and RPS of the given model and test data

        Parameters:
            model : model with .predict_proba() and .predict() methods
            X_test (DataFrame) : test sample features \n
            y_test (DataFrame) : test sample targets \n
    """
    y_pred = model.predict(X_test)

    # Confusion Matrix
    conf_matr = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=['Home Win', 'Draw', 'Away Win'])
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()

    # Precision & Recall
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['Home Win', 'Draw', 'Away Win'],
                                zero_division=0, digits=4))

    # RPS
    print("RPS: ", RPS_count(model, X_test, y_test))


def train_dummy_most_frequent(X_train, y_train, random_state=42):
    """
    Fits most frequent classifier to train data
    """
    dc_mf = DummyClassifier(strategy="most_frequent", random_state=random_state)
    dc_mf.fit(X_train, y_train)
    return dc_mf


def train_dummy_stratified(X_train, y_train, random_state=42):
    """
    Fits stratified classifier to train data
    """
    dc_mf = DummyClassifier(strategy="stratified", random_state=random_state)
    dc_mf.fit(X_train, y_train)
    return dc_mf


def train_basic_random_forest(X_train, y_train, random_state=42):
    """
    Fits default sklearn random forest classifier to train data
    """
    rfc = RandomForestClassifier(random_state=random_state)
    rfc.fit(X_train, y_train)
    return rfc


def train_rf_grid(X_train, y_train, X_val, y_val, param_grid, n_jobs=4, random_state=42):
    """
    Trains random forest using grid search to choose the best hyperparameters set from param_grid
    """
    rfc = RandomForestClassifier(random_state=random_state)
    train_indices = np.full((X_train.shape[0],), -1, dtype=int)
    test_indices = np.full((X_val.shape[0],), 0, dtype=int)
    test_fold = np.append(train_indices, test_indices)
    ps = PredefinedSplit(test_fold)
    cv_rfc = GridSearchCV(estimator=rfc, scoring='accuracy', param_grid=param_grid, verbose=3,
                          cv=ps, n_jobs=n_jobs)
    if type(X_train) == pd.DataFrame:
        X = pd.concat([X_train, X_val], axis=0)
    else:
        X = vstack([X_train, X_val])
    cv_rfc.fit(X, pd.concat([y_train, y_val], axis=0))
    return cv_rfc

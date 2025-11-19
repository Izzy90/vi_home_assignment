import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(
    X,
    y,
    *,
    test_size: float = 0.5,
    random_state: int = 42,
):
    """Train an XGBoost classifier and return the model plus evaluation metrics."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = xgboost.XGBClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred)

    return clf, {"roc_auc": roc_auc, "y_pred": y_pred, "y_val": y_val}



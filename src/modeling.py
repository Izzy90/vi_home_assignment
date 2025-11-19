import xgboost
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


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

    # generate roc_auc
    roc_auc = roc_auc_score(y_val, y_pred)

    # generate precision-recall curve and save it to a file
    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.savefig("precision_recall_curve.png")
    plt.close()

    # Compute F1 scores for all thresholds and pick the optimal one
    thresholds = _
    f1_scores = []
    for thresh in thresholds:
        y_pred_bin = (y_pred >= thresh).astype(int)
        f1_scores.append(f1_score(y_val, y_pred_bin))
    # Exclude first threshold since it's always inf and gives 0
    if len(thresholds) > 1:
        best_index = f1_scores[1:].index(max(f1_scores[1:])) + 1
    else:
        best_index = 0
    optimal_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    # Define classifier with optimal threshold
    def optimal_threshold_classifier(pred_proba):
        return (pred_proba >= optimal_threshold).astype(int)

    
    
    return clf, {"roc_auc": roc_auc, "y_pred": y_pred, "y_val": y_val}



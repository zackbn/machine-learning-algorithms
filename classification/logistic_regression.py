"""Régression logistique sur Breast Cancer : matrice de confusion, rapport et courbe ROC."""
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import evaluate, load_data, save_figure


def build_model():
    # Sans standardisation, l'optimiseur ne converge pas sur ces 30 variables.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def main():
    X_train, X_test, y_train, y_test, classes = load_data("breast_cancer")
    model = build_model().fit(X_train, y_train)
    evaluate(model, X_test, y_test, classes, "Régression logistique", "clf_logreg_confusion.png")

    # Courbe ROC : probabilité de la classe positive (1 = bénigne)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)   # l'ordre est bien FPR, TPR, seuils
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC : {auc:.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"Régression logistique (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Courbe ROC")
    ax.legend()
    save_figure(fig, "clf_logreg_roc.png")
    plt.show()


if __name__ == "__main__":
    main()

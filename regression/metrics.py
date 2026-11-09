"""Métriques de régression implémentées à la main avec NumPy.

Elles sont vérifiées contre scikit-learn dans utils.evaluate().
"""
import numpy as np


def mean_absolute_error(y_true, y_pred):
    """MAE : erreur absolue moyenne."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """MSE : erreur quadratique moyenne."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """RMSE : racine de la MSE, dans la même unité que la cible."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE : erreur absolue moyenne en pourcentage."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def r2_score(y_true, y_pred):
    """R² : part de la variance expliquée par le modèle."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

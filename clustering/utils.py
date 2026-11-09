"""Fonctions partagées par les scripts de clustering."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"

MALL_CSV = DATA_DIR / "Mall_Customers.csv"
# Colonnes 3 et 4 du fichier : revenu annuel (k$) et score de dépenses (1-100)
MALL_FEATURES = [3, 4]


def load_mall_customers(path: Path = MALL_CSV):
    """Charge le dataset Mall Customers et renvoie (X, noms_des_colonnes)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Téléchargez Mall_Customers.csv (voir data/README.md) "
            "et placez-le dans le dossier data/."
        )
    df = pd.read_csv(path)
    X = df.iloc[:, MALL_FEATURES].to_numpy(dtype=float)
    return X, list(df.columns[MALL_FEATURES])


def save_figure(fig, name: str):
    """Enregistre une figure dans images/."""
    IMAGES_DIR.mkdir(exist_ok=True)
    out = IMAGES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Figure enregistrée : {out.relative_to(ROOT)}")

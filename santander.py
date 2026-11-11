"""Fonctions communes au notebook et à pipeline.py (sujet 07 — transaction client).

Tout ce qui apprend quelque chose de la cible (sélection de variables, binning, modèle)
est appris DANS un pli d'entraînement, jamais sur la ligne évaluée : pas de fuite.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CIBLE = "target"
ID = "ID_code"
GRAINE = 42


# --------------------------------------------------------------------------- #
# Chargement                                                                   #
# --------------------------------------------------------------------------- #
def charger(chemin: str) -> pd.DataFrame:
    """Lit un csv du sujet en float32 (2x moins de mémoire, aucune perte utile)."""
    entete = pd.read_csv(chemin, nrows=0).columns
    types = {c: "float32" for c in entete if c.startswith("var_")}
    return pd.read_csv(chemin, dtype=types)


def colonnes_var(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("var_")]


# --------------------------------------------------------------------------- #
# Plancher : la règle sans modèle                                              #
# --------------------------------------------------------------------------- #
class RegleUneVariable(BaseEstimator, ClassifierMixin):
    """Règle « sans modèle » : trier les clients sur UNE variable brute.

    La variable et son sens (croissant / décroissant) sont choisis sur le pli
    d'entraînement uniquement, puis appliqués tels quels.
    """

    def fit(self, X, y):
        X = np.asarray(X)
        aucs = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
        force = np.abs(aucs - 0.5)
        self.j_ = int(np.argmax(force))
        self.signe_ = 1.0 if aucs[self.j_] >= 0.5 else -1.0
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.signe_ * np.asarray(X)[:, self.j_]

    def predict_proba(self, X):
        s = self.decision_function(X)
        r = (s.argsort().argsort() + 0.5) / len(s)  # rang -> pseudo-probabilité
        return np.c_[1 - r, r]


# --------------------------------------------------------------------------- #
# Notre modèle : Bayes naïf non paramétrique (log-rapports par classes de     #
# quantiles, une variable à la fois, puis somme).                              #
# --------------------------------------------------------------------------- #
class BayesNaifQuantiles(BaseEstimator, ClassifierMixin):
    """Pour chaque variable : découpe en `n_bins` intervalles de quantiles, mesure
    dans chaque intervalle le log-rapport  log P(x|1)/P(x|0)  (lissé), puis additionne
    les variables. C'est un Bayes naïf dont la densité n'est PAS supposée gaussienne.

    Il exploite le fait mesuré sur cette table : les variables sont quasi
    indépendantes entre elles, et leur lien avec la cible n'est pas monotone.
    """

    def __init__(self, n_bins: int = 64, lissage: float = 50.0, voisinage: float = 2.0):
        self.n_bins = n_bins
        self.lissage = lissage        # poids du taux global dans chaque tranche (prior bêta)
        self.voisinage = voisinage    # écart-type, en tranches, du lissage entre tranches voisines

    @staticmethod
    def _lisser(v, sigma):
        if not sigma:
            return v
        r = int(np.ceil(3 * sigma))
        noyau = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        noyau /= noyau.sum()
        return np.convolve(np.pad(v, r, mode="edge"), noyau, mode="valid")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        pi = y.mean()
        self.prior_ = np.log(pi / (1 - pi))
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.bords_, self.tables_ = [], []
        for j in range(X.shape[1]):
            bords = np.unique(np.quantile(X[:, j], qs))
            b = np.searchsorted(bords, X[:, j], side="right")
            k = len(bords) + 1
            c1 = self._lisser(np.bincount(b, weights=y, minlength=k), self.voisinage)
            n = self._lisser(np.bincount(b, minlength=k).astype(float), self.voisinage)
            # taux de transaction de la tranche, tiré vers le taux global : une tranche rare ne crie pas
            r = (c1 + self.lissage * pi) / (n + self.lissage)
            self.bords_.append(bords)
            self.tables_.append((np.log(r / (1 - r)) - self.prior_).astype(np.float32))
        self.classes_ = np.array([0, 1])
        return self

    def contributions(self, X):
        """Matrice (n, p) des log-rapports par variable : sert à l'analyse d'erreurs."""
        X = np.asarray(X, dtype=np.float32)
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            out[:, j] = self.tables_[j][np.searchsorted(self.bords_[j], X[:, j], side="right")]
        return out

    def decision_function(self, X):
        return self.contributions(X).sum(axis=1) + self.prior_

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.c_[1 - p, p]


class SelectionAUC(BaseEstimator):
    """Garde les k variables dont l'AUC seule (|AUC-0,5|) est la plus forte,
    mesurée sur les données passées à fit — donc sur le pli d'entraînement."""

    def __init__(self, k: int = 200):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)
        f = np.abs(np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])]) - 0.5)
        self.garde_ = np.sort(np.argsort(-f)[: self.k])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.garde_]


# --------------------------------------------------------------------------- #
# Protocole d'essai                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Journal:
    """Tableau des essais : chaque essai, réussi ou raté, y laisse une ligne."""

    lignes: list = field(default_factory=list)

    def noter(self, nom, auc_cv, ecart, auc_test=np.nan, secondes=np.nan, verdict="", note=""):
        self.lignes.append(dict(essai=nom, auc_cv=round(auc_cv, 4), ecart_type=round(ecart, 4),
                                auc_test_interne=round(auc_test, 4) if auc_test == auc_test else np.nan,
                                secondes=round(secondes, 1), verdict=verdict, note=note))

    def tableau(self) -> pd.DataFrame:
        return pd.DataFrame(self.lignes)


def score_cv(fabrique, X, y, n_plis=5, graine=GRAINE, retour_oof=False):
    """AUC moyenne en validation croisée stratifiée. `fabrique()` renvoie un modèle neuf
    à chaque pli : tout ce qu'il apprend vient du pli d'entraînement."""
    X = np.asarray(X)
    y = np.asarray(y)
    plis = StratifiedKFold(n_plis, shuffle=True, random_state=graine)
    oof = np.zeros(len(y))
    aucs = []
    t = time.perf_counter()
    for tr, va in plis.split(X, y):
        m = fabrique().fit(X[tr], y[tr])
        s = score(m, X[va])
        oof[va] = s
        aucs.append(roc_auc_score(y[va], s))
    sec = time.perf_counter() - t
    if retour_oof:
        return float(np.mean(aucs)), float(np.std(aucs)), sec, oof
    return float(np.mean(aucs)), float(np.std(aucs)), sec


def score(modele, X):
    """Score de classement (seul l'ordre compte pour l'AUC)."""
    if hasattr(modele, "decision_function"):
        return modele.decision_function(X)
    return modele.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# Décision : du score au oui / non                                             #
# --------------------------------------------------------------------------- #
def seuil_cout_minimal(y, p, cout_fp, cout_fn):
    """Seuil de probabilité qui minimise le coût total  FP*cout_fp + FN*cout_fn."""
    ordre = np.argsort(-p)
    ys = np.asarray(y)[ordre]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    cout = fp * cout_fp + fn * cout_fn
    i = int(np.argmin(cout))
    return float(p[ordre][i]), float(cout[i]), int(i + 1)

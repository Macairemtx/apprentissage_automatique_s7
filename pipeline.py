"""Pipeline exécutable : train.csv + a_decider.csv  ->  sorties/decision.csv (ID_code,target).

    python pipeline.py                      # chemins par défaut (dossier donnees/)
    python pipeline.py --train T.csv --decider D.csv --sortie sorties/decision.csv

Modèle : Bayes naïf non paramétrique (log-rapports par tranches de quantiles), k variables choisies
sur l'AUC seule. La part de clients sollicités est celle qui minimise le coût en validation croisée.
scikit-learn seul, CPU, < 10 minutes (mesuré : voir README).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

import santander as S

# --- configuration retenue (issue du notebook) ---------------------------------
N_BINS = 64
K_VARIABLES = 200
COUT_FP, COUT_FN = 2.0, 20.0
N_PLIS = 5


def fabrique():
    return make_pipeline(S.SelectionAUC(K_VARIABLES), S.BayesNaifQuantiles(n_bins=N_BINS))


def main():
    ici = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=ici / "donnees" / "train.csv")
    ap.add_argument("--decider", default=ici / "donnees" / "a_decider.csv")
    ap.add_argument("--sortie", default=ici / "sorties" / "decision.csv")
    a = ap.parse_args()

    t0 = time.perf_counter()
    train, decider = S.charger(a.train), S.charger(a.decider)
    vars_ = S.colonnes_var(train)
    X, y = train[vars_].to_numpy(np.float32), train[S.CIBLE].to_numpy(int)
    print(f"chargé : {len(y)} clients connus, {len(decider)} à décider ({time.perf_counter()-t0:.1f} s)")

    # 1) part de clients à solliciter : minimise le coût sur les scores hors-pli
    auc, sd, _, oof = S.score_cv(fabrique, X, y, N_PLIS, retour_oof=True)
    _, cout, n_sel = S.seuil_cout_minimal(y, oof, COUT_FP, COUT_FN)
    part = n_sel / len(y)
    print(f"AUC hors-pli {auc:.4f} ± {sd:.4f} ; part sollicitée {part:.1%} ; coût {cout/len(y):.3f} €/client")

    # 2) modèle final sur toutes les lignes connues, décision sur les 40 000
    modele = fabrique().fit(X, y)
    s = S.score(modele, decider[vars_].to_numpy(np.float32))
    decision = np.zeros(len(s), int)
    decision[np.argsort(-s)[: int(round(part * len(s)))]] = 1

    sortie = Path(a.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({S.ID: decider[S.ID], S.CIBLE: decision}).to_csv(sortie, index=False)
    print(f"{decision.sum()} clients sollicités sur {len(decision)} -> {sortie}")
    print(f"durée totale : {time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()

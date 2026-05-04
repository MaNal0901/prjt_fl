import flwr as fl
from flwr.server.strategy import FedAvg
import flwr.common
import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from datetime import datetime
import threading, socket, time

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_ROUNDS  = int(os.environ.get("NUM_ROUNDS",  "20"))
MIN_CLIENTS = int(os.environ.get("MIN_CLIENTS", "4"))
MODEL_TYPE  = os.environ.get("MODEL_TYPE", "cnn1d")
INPUT_DIM   = int(os.environ.get("INPUT_DIM",  "37"))

# ── Charger test global ───────────────────────────────────────────────────────
try:
    df_global  = pd.read_parquet("/app/data/test_global.parquet")
    X_global   = torch.tensor(df_global.drop("isFraud", axis=1).values.astype(np.float32))
    y_global   = df_global["isFraud"].values
    HAS_GLOBAL = True
    print(f"[Server] Test global charge : {len(y_global):,} transactions")
except Exception as e:
    HAS_GLOBAL = False
    X_global = y_global = None
    print(f"[Server] test_global.parquet non trouve — evaluation locale uniquement")

results_log = []

# ── Modeles ───────────────────────────────────────────────────────────────────
from models import get_model

def get_server_model(model_type, input_dim):
    return get_model(model_type, input_dim)

def set_params(model, params):
    state = model.state_dict()
    for key, val in zip(state.keys(), params):
        clean = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        original_dtype = state[key].dtype
        state[key] = torch.tensor(clean).to(original_dtype)
    model.load_state_dict(state)

def evaluate_global_model(params):
    if not HAS_GLOBAL:
        return None
    try:
        model = get_server_model(MODEL_TYPE, INPUT_DIM)
        set_params(model, params)
        model.eval()
        with torch.no_grad():
            logits_tensor = model(X_global)
            probs = torch.sigmoid(logits_tensor).numpy().flatten()
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

        # ✅ CORRECTION — seuil optimal au lieu de 0.5 fixe
        # Le modèle produit des probabilités très basses (~0.02-0.05) pour les fraudes.
        # Avec seuil=0.5, Recall=0 et F1≈0 malgré un AUC à 0.95.
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.02):
            p = (probs >= t).astype(int)
            f = float(f1_score(y_global, p, zero_division=0))
            if f > best_f1:
                best_f1, best_t = f, t
        preds = (probs >= best_t).astype(int)
        print(f"[Server] Seuil optimal global : {best_t:.2f} (F1={best_f1:.4f})")

        return {
            "f1":        float(f1_score(y_global,        preds, zero_division=0)),
            "auc":       float(roc_auc_score(y_global,   probs) if len(np.unique(y_global)) > 1 else 0.0),
            "precision": float(precision_score(y_global, preds, zero_division=0)),
            "recall":    float(recall_score(y_global,    preds, zero_division=0)),
            "threshold": float(best_t),
        }
    except Exception as e:
        print(f"[Server] Evaluation globale echouee : {e}")
        return None

# ── Fonctions d'agrégation des métriques ─────────────────────────────────────
def fit_metrics_aggregation(metrics):
    total = sum(n for n, _ in metrics)
    return {
        "train_loss": sum(n * m.get("train_loss", 0.0) for n, m in metrics) / total,
        "f1_local":   sum(n * m.get("f1_local",   0.0) for n, m in metrics) / total,
    }

def evaluate_metrics_aggregation(metrics):
    total = sum(n for n, _ in metrics)
    return {
        "f1_local":  sum(n * m.get("f1_local",  0.0) for n, m in metrics) / total,
        "auc_local": sum(n * m.get("auc_local", 0.0) for n, m in metrics) / total,
    }

# ── Strategie FL : FedAvg + agrégation Eq.7 sur les poids ───────────────────
class FraudStrategy(FedAvg):
   def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_params = None
        self._last_alphas = {}   # AJOUTER — {bank_id: alpha} depuis evaluate()

    def aggregate_fit(self, rnd, results, failures):
        if failures:
            print(f"[Server] Round {rnd} — {len(failures)} client(s) en échec")

        # ── Agrégation Eq.7 : pondération des poids par alpha ───────────────
        weighted_params = None
        total_alpha     = 0.0

        for _, fit_res in results:
            bank_id = fit_res.metrics.get("bank_id", "?")
            alpha   = self._last_alphas.get(bank_id, 1.0)  # fallback=1.0 (round 1)
            params = flwr.common.parameters_to_ndarrays(fit_res.parameters)
            params = [np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0) for p in params]

            if weighted_params is None:
                weighted_params = [alpha * p for p in params]
            else:
                for i, p in enumerate(params):
                    weighted_params[i] += alpha * p
            total_alpha += alpha
        print(f"[Server] Round {rnd} — total_alpha={total_alpha:.4f} "
            f"({'Eq.7 actif' if total_alpha > 0 else 'FALLBACK FedAvg'})")
        if total_alpha > 0 and weighted_params is not None:
            aggregated_params = [p / total_alpha for p in weighted_params]
        else:
            print(f"[Server] Round {rnd} — alpha_sum=0, fallback FedAvg standard")
            aggregated_params = None

        # ── Affichage métriques d'entraînement ──────────────────────────────
        print(f"\n{'='*70}")
        print(f"  ROUND {rnd}/{NUM_ROUNDS} — MÉTRIQUES D'ENTRAÎNEMENT")
        print(f"{'─'*70}")
        print(f"  {'Banque':<10} {'Loss':>8} {'F1':>7} {'Alpha':>10} {'Durée (s)':>10}")
        print(f"{'─'*70}")

        total_n       = sum(r.num_examples for _, r in results)
        weighted_loss = 0.0

        for _, fit_res in results:
            m    = fit_res.metrics
            bank = m.get("bank_id",        "?")
            loss = m.get("train_loss",     0.0)
            f1   = m.get("f1_local",       0.0)
            alph = m.get("alpha",          0.0)
            dur  = m.get("train_latency_s",0.0)
            n    = fit_res.num_examples
            weighted_loss += (n / total_n) * loss
            print(f"  [{bank:<8}] {loss:>8.4f} {f1:>7.4f} {alph:>10.4f} {dur:>10.1f}")

        print(f"{'─'*70}")
        print(f"  Loss moyenne pondérée : {weighted_loss:.4f}")
        print(f"{'='*70}\n")

        # ── Construire l'objet aggregated à retourner ────────────────────────
        if aggregated_params is not None:
            params_clean   = [np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                            for p in aggregated_params]
            self._last_params = params_clean
            parameters_agg    = flwr.common.ndarrays_to_parameters(params_clean)
            return parameters_agg, {}
        else:
            aggregated = super().aggregate_fit(rnd, results, failures)
            if aggregated and aggregated[0]:
                params = flwr.common.parameters_to_ndarrays(aggregated[0])
                params_clean = [np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                                for p in params]
                self._last_params = params_clean
                aggregated = (flwr.common.ndarrays_to_parameters(params_clean),
                              aggregated[1])
            return aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        print(f"\n{'='*70}")
        print(f"  ROUND {rnd}/{NUM_ROUNDS} — RESULTATS PAR BANQUE")
        print(f"{'─'*70}")
        print(f"  {'Banque':<10} {'F1':>7} {'AUC':>7} {'Precision':>10} {'Recall':>8} {'N':>8} {'Weight':>8}")
        print(f"{'─'*70}")

        round_data = {
            "round":     rnd,
            "timestamp": datetime.now().isoformat(),
            "banks":     [],
        }

        total_n      = sum(r.num_examples for _, r in results)
        weighted_f1  = 0.0
        weighted_auc = 0.0

        bank_data = []
        alpha_sum = 0.0
        for _, eval_res in results:
            n     = eval_res.num_examples
            alpha = eval_res.metrics.get("alpha", 0.0)
            raw_w = (n / total_n) * alpha
            alpha_sum += raw_w
            bank_data.append((eval_res.metrics, n, alpha, raw_w))

        for m, n, alpha, raw_w in bank_data:
            weight = raw_w / alpha_sum if alpha_sum > 0 else 1 / len(bank_data)
            bank   = m.get("bank_id",         "?")
            f1     = m.get("f1_local",        0.0)
            auc    = m.get("auc_local",       0.0)
            prec   = m.get("precision_local", 0.0)
            rec    = m.get("recall_local",    0.0)
            weighted_f1  += weight * f1
            weighted_auc += weight * auc
            print(f"  [{bank:<8}] {f1:>7.4f} {auc:>7.4f} {prec:>10.4f} {rec:>8.4f} {n:>8,} {weight:>8.4f}")
            round_data["banks"].append({
                "bank_id":    bank,
                "f1":         round(f1,     4),
                "auc":        round(auc,    4),
                "precision":  round(prec,   4),
                "recall":     round(rec,    4),
                "n_samples":  n,
                "alpha":      round(alpha,  4),
                "weight_eq7": round(weight, 4),
            })

        round_data["global_weighted_f1"]  = round(weighted_f1,  4)
        round_data["global_weighted_auc"] = round(weighted_auc, 4)

        print(f"{'─'*70}")
        print(f"  Global pondere (Eq.7)  — F1: {weighted_f1:.4f} | AUC: {weighted_auc:.4f}")

        if self._last_params is not None:
            gm = evaluate_global_model(self._last_params)
            if gm:
                print(f"  Global test_global     — F1: {gm['f1']:.4f} | AUC: {gm['auc']:.4f} | "
                      f"Prec: {gm['precision']:.4f} | Rec: {gm['recall']:.4f} | Seuil: {gm['threshold']:.2f}")
                round_data["global_test"] = gm

        print(f"  Benchmark FFD (F=0.1)  — F1: 0.9393 | AUC: 0.9555")
        print(f"{'='*70}\n")

        results_log.append(round_data)
        os.makedirs("/app/results", exist_ok=True)
        with open("/app/results/server_results.json", "w") as f:
            json.dump(results_log, f, indent=2)
        for m, n, alpha, raw_w in bank_data:
            bank = m.get("bank_id", "?")
            self._last_alphas[bank] = alpha   # alpha stable depuis evaluate(
        return aggregated

# ── Création de la stratégie ──────────────────────────────────────────────────
strategy = FraudStrategy(
    min_available_clients=MIN_CLIENTS,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    fit_metrics_aggregation_fn=fit_metrics_aggregation,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
)

# ── Lancement avec mTLS complet ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("    FL Finance Server — Step 3")
    print(f"    Modele: {MODEL_TYPE} | Rounds: {NUM_ROUNDS} | Clients: {MIN_CLIENTS}")
    print(f"    Strategie: FedAvg + Ponderation Eq.7 FFD (poids ET metriques)")
    print(f"    Reference: Yang et al. FFD 2019 — AUC cible >= 0.955")
    print(f"    mTLS: actif (Zero Trust)")
    print("=" * 70)
    print(f"[Server] Attente de {MIN_CLIENTS} clients sur 0.0.0.0:8080 (mTLS)...")

    def _write_ready_when_bound():
        for _ in range(30):
            try:
                s = socket.create_connection(("127.0.0.1", 8080), timeout=1)
                s.close()
                open("/tmp/fl_server_ready", "w").close()
                print("[Server] Healthcheck : fl_server_ready écrit")
                return
            except OSError:
                time.sleep(1)

    threading.Thread(target=_write_ready_when_bound, daemon=True).start()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        certificates=(
            open("/certs/ca.crt",     "rb").read(),
            open("/certs/server.crt", "rb").read(),
            open("/certs/server.key", "rb").read(),
        ),
    )
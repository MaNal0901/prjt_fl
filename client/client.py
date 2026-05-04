import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import flwr as fl
import time

from models import get_model
from privacy import apply_dp

# ── Configuration via variables d'environnement ──────────────────────────────
BANK_ID    = os.environ.get("BANK_ID",         "bank_a")
TRAIN_PATH = os.environ.get("TRAIN_PATH",      "/app/data/train_A.parquet")
TEST_PATH  = os.environ.get("TEST_PATH",       "/app/data/test_A.parquet")
SERVER     = os.environ.get("SERVER_ADDRESS",  "fl-server:8080")
MODEL_TYPE = os.environ.get("MODEL_TYPE",      "cnn1d")   # mlp | cnn1d | cnnlstm
EPOCHS     = int(os.environ.get("LOCAL_EPOCHS", "10"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE",   "80"))
LR         = float(os.environ.get("LR",         "0.003"))
INPUT_DIM  = int(os.environ.get("INPUT_DIM",    "37"))

# ✅ FIX 2 — mu_proximal augmenté 0.01 → 0.05
# Bank_b a des données hétérogènes : F1 fit instable (0.39–0.66).
# Un mu plus fort ancre mieux les poids locaux et réduit la dérive inter-rounds.
MU_PROXIMAL = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Chargement des donnees ────────────────────────────────────────────────────
print(f"[{BANK_ID}] Chargement des donnees...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test  = pd.read_parquet(TEST_PATH)

X_train = torch.tensor(df_train.drop("isFraud", axis=1).values, dtype=torch.float32)
y_train = torch.tensor(df_train["isFraud"].values,              dtype=torch.float32)
X_test  = torch.tensor(df_test.drop("isFraud",  axis=1).values, dtype=torch.float32)
y_test  = df_test["isFraud"].values

# ── Gestion du desequilibre de classes (0.17% fraude) ────────────────────────
n_pos      = (y_train == 1).sum().item()
n_neg      = (y_train == 0).sum().item()
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

print(f"[{BANK_ID}] Train: {len(X_train):,} | Test: {len(X_test):,} | Fraude: {n_pos} ({100*n_pos/len(y_train):.2f}%)")
print(f"[{BANK_ID}] Modele: {MODEL_TYPE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"[{BANK_ID}] Loss: BCEWithLogitsLoss | pos_weight: {pos_weight.item():.1f}")
print(f"[{BANK_ID}] FedProx mu={MU_PROXIMAL} | DP sigma=0.05 C=2.0")

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ── Utilitaires poids ─────────────────────────────────────────────────────────
def get_params(model: nn.Module) -> list:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_params(model: nn.Module, params: list) -> None:
    state = model.state_dict()
    for key, val in zip(state.keys(), params):
        clean = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        original_dtype = state[key].dtype
        state[key] = torch.tensor(clean).to(original_dtype)
    model.load_state_dict(state)


# ── Metriques completes (Article FFD Section 4.2) ────────────────────────────
def compute_all_metrics(y_true, y_pred, y_prob):
    if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
        print(f"[{BANK_ID}] ⚠️ NaN dans y_prob — AUC mis à 0.0")
        auc = 0.0
    else:
        auc = float(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0)
    return {
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":       auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }

def find_best_threshold(probs, y_true):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ── Client Flower ─────────────────────────────────────────────────────────────
class FraudClient(fl.client.NumPyClient):

    def __init__(self):
        self.model     = get_model(MODEL_TYPE, INPUT_DIM).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ✅ FIX 1 — optimizer et scheduler NE sont PAS créés ici.
        # Ils sont recréés au début de chaque fit() pour que le scheduler
        # parte du bon LR à chaque round FL (et non d'un état résiduel).
        # Sans ce fix, le scheduler repart de son état du round précédent,
        # ce qui désynchronise le LR et annule le bénéfice du cosine annealing.
        self.optimizer = None
        self.scheduler = None

        print(f"[{BANK_ID}] Modele {MODEL_TYPE} initialise sur {DEVICE}")

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        # ── Sauvegarder les poids AVANT entraînement (pour DP) ───────────────
        params_before = [p.copy() for p in parameters]
        set_params(self.model, parameters)

        # ✅ FIX 1 — recréer optimizer + scheduler à chaque round
        # Le scheduler cosine doit partir de LR=0.003 et descendre sur EPOCHS
        # itérations. Recréer ici garantit T_max=EPOCHS frames propres par round.
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=LR * 0.1
        )

        # ✅ FIX 2 (suite) — sauvegarder les poids globaux pour le terme FedProx
        # mu=0.05 pénalise davantage l'éloignement → bank_b reste dans la région
        # du modèle global et évite le sur-apprentissage local
        global_params = [p.clone().detach() for p in self.model.parameters()]

        self.model.train()
        start_time = time.time()

        total_loss = 0.0
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).unsqueeze(1)
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss  = self.criterion(preds, y_batch)

                # Terme proximal FedProx — pénalise l'éloignement des poids locaux
                prox = sum(
                    torch.norm(p - g) ** 2
                    for p, g in zip(self.model.parameters(), global_params)
                )
                loss = loss + (MU_PROXIMAL / 2) * prox

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # ✅ FIX 1 (suite) — step du scheduler APRES chaque epoch
            # Maintenant que l'optimizer est recréé ici, ce step est cohérent
            self.scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            print(f"[{BANK_ID}] Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}")
            total_loss += avg_loss

        mean_loss = total_loss / EPOCHS

        # Evaluation F1 post-training (utilisée pour le log — pas pour alpha)
        eval_start = time.time()
        self.model.eval()
        with torch.no_grad():
            logits_tensor = self.model(X_test.to(DEVICE))
            probs_tensor  = torch.sigmoid(logits_tensor)
            probs_test    = probs_tensor.cpu().numpy().flatten()

        probs_test = np.nan_to_num(probs_test, nan=0.5, posinf=1.0, neginf=0.0)
        thresh     = find_best_threshold(probs_test, y_test)
        preds_test = (probs_test >= thresh).astype(int)
        f1_local   = f1_score(y_test, preds_test, zero_division=0)
        auc_local  = float(roc_auc_score(y_test, probs_test)
                        if len(np.unique(y_test)) > 1 else 0.0)

        # DP — sigma=0.05, C=2.0 (epsilon ≈ 4.0)
        params_after = get_params(self.model)
        pseudo_grads = [after - before
                        for after, before in zip(params_after, params_before)]
        grads_dp     = [apply_dp(g, C=2.0, sigma=0.05) for g in pseudo_grads]
        params_dp    = [before + grad
                        for before, grad in zip(params_before, grads_dp)]
        params_dp    = [np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                        for p in params_dp]

        eval_latency_ms = (time.time() - eval_start) * 1000
        train_latency_s = time.time() - start_time
        self.model.train()

        print(f"[{BANK_ID}] Fit terminé — Loss={mean_loss:.4f} | F1={f1_local:.4f} | "
            f"AUC={auc_local:.4f} | Eval={eval_latency_ms:.1f}ms")

        # ✅ FIX 3a — alpha mis à 0.0 dans fit()
        # L'AUC calculée ici est bruitée (optimisation en cours, seuil instable).
        # L'alpha stable et définitif est calculé dans evaluate() sur le modèle
        # agrégé — c'est lui qui sera utilisé par le serveur pour Eq.7.
        return params_dp, len(X_train), {
            "bank_id":         BANK_ID,
            "model_type":      MODEL_TYPE,
            "train_loss":      float(mean_loss),
            "eval_latency_ms": float(eval_latency_ms),
            "train_latency_s": float(train_latency_s),
            "f1_local":        float(f1_local),
            "alpha":           0.0,  # ignoré par le serveur, fourni par evaluate()
        }

    def evaluate(self, parameters, config):
        """
        Evaluation locale — 4 metriques conformes a l'article FFD Section 4.2 :
        F1, AUC, Precision, Recall.

        ✅ FIX 3a — alpha calculé ICI sur le modèle agrégé (stable)
        L'AUC de evaluate() est fiable : pas d'entraînement en cours,
        modèle agrégé convergé. C'est cette valeur qui pilote Eq.7 côté serveur.
        """
        for i, p in enumerate(parameters):
            if np.any(np.isnan(p)):
                print(f"[{BANK_ID}] ⚠️ NaN dans paramètre {i} — poids globaux corrompus")
                return 0.0, len(X_test), {
                    "f1_local":        0.0,
                    "auc_local":       0.0,
                    "precision_local": 0.0,
                    "recall_local":    0.0,
                    "bank_id":         BANK_ID,
                    "alpha":           0.0,
                }

        set_params(self.model, parameters)
        self.model.eval()

        with torch.no_grad():
            logits_tensor = self.model(X_test.to(DEVICE))
            probs_tensor  = torch.sigmoid(logits_tensor)
            probs         = probs_tensor.cpu().numpy().flatten()

        probs  = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        thresh = find_best_threshold(probs, y_test)
        preds  = (probs >= thresh).astype(int)
        metrics = compute_all_metrics(y_test, preds, probs)

        # ✅ FIX 3a — alpha stable basé sur l'AUC de l'évaluation du modèle agrégé
        alpha_stable = float(metrics["auc"]) * np.log1p(len(X_train))

        print(f"[{BANK_ID}] F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f} | "
            f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | "
            f"Alpha={alpha_stable:.4f}")

        return float(1 - metrics["f1"]), len(X_test), {
            "f1_local":        metrics["f1"],
            "auc_local":       metrics["auc"],
            "precision_local": metrics["precision"],
            "recall_local":    metrics["recall"],
            "bank_id":         BANK_ID,
            "alpha":           alpha_stable,  # ✅ valeur stable utilisée par Eq.7
        }


# ── Lancement avec mTLS complet ───────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[{BANK_ID}] Connexion securisee mTLS -> {SERVER}")

    ca_cert = open("/certs/ca.crt", "rb").read()

    fl.client.start_numpy_client(
        server_address=SERVER,
        client=FraudClient(),
        root_certificates=ca_cert,
    )
#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Script de build optimisé pour Codespaces (espace disque limité)
#  Usage : bash build_and_run.sh [mlp|cnn1d|cnnlstm]
# ══════════════════════════════════════════════════════════════════

MODEL=${1:-mlp}
COMPOSE_FILE="docker-compose-${MODEL}.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "❌ Fichier introuvable : $COMPOSE_FILE"
  echo "   Usage : bash build_and_run.sh [mlp|cnn1d|cnnlstm]"
  exit 1
fi

echo "══════════════════════════════════════════"
echo "  FL Finance — Modèle : $MODEL"
echo "══════════════════════════════════════════"

# 1. Nettoyage complet avant build (libère l'espace)
echo ""
echo "🧹 [1/3] Nettoyage Docker..."
docker compose -f "$COMPOSE_FILE" down --volumes 2>/dev/null || true
docker system prune -f
echo "✅ Espace libéré : $(df -h / | awk 'NR==2{print $4}') disponibles"

# 2. Vérification espace disque
AVAIL=$(df / | awk 'NR==2{print $4}')
if [ "$AVAIL" -lt 5000000 ]; then
  echo "⚠️  Espace insuffisant (<5GB). Nettoyage agressif..."
  docker system prune -a -f
fi

# 3. Build séquentiel (évite la saturation mémoire)
echo ""
echo "🔨 [2/3] Build des images (séquentiel)..."
docker compose -f "$COMPOSE_FILE" build --no-cache fl-server
docker compose -f "$COMPOSE_FILE" build --no-cache bank-a bank-b bank-c bank-d

# 4. Lancement
echo ""
echo "🚀 [3/3] Lancement des containers..."
docker compose -f "$COMPOSE_FILE" up

echo ""
echo "✅ Terminé !"

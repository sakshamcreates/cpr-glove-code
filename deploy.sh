#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Manomitra Full-Stack Deployment Helper
# =============================================================================
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh [frontend|backend|all]
#
# Requirements:
#   - Node.js v18+
#   - Vercel CLI  (npm i -g vercel)
#   - Git configured with remote 'origin'
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

log()     { echo -e "${CYAN}[deploy]${NC} $*"; }
success() { echo -e "${GREEN}[deploy] ✅ $*${NC}"; }
warn()    { echo -e "${YELLOW}[deploy] ⚠️  $*${NC}"; }
error()   { echo -e "${RED}[deploy] ❌ $*${NC}"; exit 1; }

# ── Root detection ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_DIR="$SCRIPT_DIR/backend"

check_dirs() {
  [[ -d "$FRONTEND_DIR" ]] || error "frontend/ directory not found."
  [[ -d "$BACKEND_DIR" ]]  || error "backend/ directory not found."
}

# ── Dependency check ──────────────────────────────────────────────────────────
check_deps() {
  log "Checking dependencies..."
  command -v node  >/dev/null 2>&1 || error "node not found. Install Node.js v18+."
  command -v npm   >/dev/null 2>&1 || error "npm not found."
  command -v git   >/dev/null 2>&1 || error "git not found."
  NODE_VER=$(node -v | sed 's/v//')
  MAJOR="${NODE_VER%%.*}"
  [[ "$MAJOR" -ge 18 ]] || error "Node.js v18+ required (found v${NODE_VER})."
  success "Node.js v${NODE_VER} — OK"
}

# ── Frontend — build & Vercel deploy ─────────────────────────────────────────
deploy_frontend() {
  log "Building frontend (React 19 + Vite)..."
  cd "$FRONTEND_DIR"

  [[ -f ".env" ]] || warn ".env not found in frontend/ — using defaults."

  npm ci --silent
  npm run build

  success "Frontend built → frontend/dist/"

  if command -v vercel >/dev/null 2>&1; then
    log "Deploying frontend to Vercel..."
    vercel --prod --yes
    success "Frontend deployed to Vercel."
  else
    warn "Vercel CLI not installed. Run: npm i -g vercel"
    warn "Then manually run: cd frontend && vercel --prod"
  fi

  cd "$SCRIPT_DIR"
}

# ── Backend — push to trigger Render auto-deploy ──────────────────────────────
deploy_backend() {
  log "Preparing backend for Render deployment..."
  cd "$BACKEND_DIR"

  [[ -f ".env" ]] || warn ".env not found in backend/ — ensure Render env vars are set."

  npm ci --silent
  success "Backend dependencies verified."

  log "Render deploys automatically on git push to main."
  log "Pushing to origin/main now..."

  cd "$SCRIPT_DIR"
  git push origin main

  success "Pushed to GitHub. Render will auto-deploy backend in ~60s."
  log "Monitor at: https://dashboard.render.com"
}

# ── Health check after deploy ─────────────────────────────────────────────────
health_check() {
  log "Running post-deploy health checks..."

  BACKEND_URL="https://YOUR-MANOMITRA-BACKEND.onrender.com/api/health"
  ML_URL="https://manomitra-ai-engine.onrender.com/docs"

  check_url() {
    local label="$1" url="$2"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$url" || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
      success "$label → HTTP $HTTP_CODE"
    else
      warn "$label → HTTP $HTTP_CODE (may still be warming up)"
    fi
  }

  check_url "Backend API  " "$BACKEND_URL"
  check_url "ML Engine    " "$ML_URL"
}

# ── Entry point ───────────────────────────────────────────────────────────────
TARGET="${1:-all}"

check_deps
check_dirs

case "$TARGET" in
  frontend)
    log "Deploying FRONTEND only..."
    deploy_frontend
    ;;
  backend)
    log "Deploying BACKEND only..."
    deploy_backend
    ;;
  all)
    log "Deploying FULL STACK (frontend + backend)..."
    deploy_frontend
    deploy_backend
    health_check
    ;;
  health)
    health_check
    ;;
  *)
    error "Unknown target '$TARGET'. Use: frontend | backend | all | health"
    ;;
esac

echo ""
success "Deployment complete. 🌸 Manomitra is live."
echo ""
echo "  🌐 Web App  : https://YOUR-MANOMITRA-APP.vercel.app"
echo "  ⚙️  Backend : https://YOUR-MANOMITRA-BACKEND.onrender.com/api/health"
echo "  🧠 ML Engine: https://manomitra-ai-engine.onrender.com/docs"
echo "  📱 WhatsApp : https://wa.me/15556680031"
echo ""

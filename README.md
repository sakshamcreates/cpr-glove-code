# Manomitra — Senior Cognitive Care Companion

Manomitra is a senior cognitive-care companion for elderly patients in North-East India, prepared for a hackathon demo:

- **App (frontend + backend):** AI-powered cognitive-care portal — patient portal, 5 cognitive games, caregiver dashboard, reminders, Memory Bank, Gemini assistant, WhatsApp reminders.
- **ML engine:** FastAPI service with adaptive difficulty and a cognitive health score.

**What Manomitra includes:**
- PWA manifest, page title, backend service name and local-storage keys under the Manomitra name.
- **Security:** the AI engine's Bhashini credentials are read from environment variables (`BHASHINI_USER_ID`, `BHASHINI_API_KEY`, `BHASHINI_INFERENCE_KEY`). Do not commit real keys.
- Frontend production build verified (`vite build` passes). AI engine endpoints verified locally (see below).

**Honest ML note — say this to judges, do not hide it:**
The AI model is a small DecisionTree trained on 500 rows of *simulated* gameplay logs (`mock_gameplay_logs.csv`). Its teacher rules are: reaction < 3.5s and 0 mistakes → level +1; reaction > 5.5s or ≥ 2 mistakes → level −1; otherwise keep level. The health score is a formula (100 − time penalty − mistake penalty), not a learned clinical model. It is a demo adaptive-difficulty engine, not a medical diagnosis tool. The backend also has a rule-based fallback if the ML service is down, and it reports `source: rule_fallback` in that case.


## System architecture

```
Patient / Caregiver (browser, React 19 PWA)
        │  VITE_API_BASE_URL (default http://localhost:5000/api)
        ▼
Node/Express backend (backend/server.js, port 5000)
  ├─ /api/patients, /api/caregivers, /api/reminders
  ├─ /api/game-sessions  →  mlService.js  →  ML_SERVICE_URL
  ├─ /api/translation, /api/speech, /api/whatsapp
  └─ node-cron reminder scheduler
        │                    │
        ▼                    ▼
  MongoDB Atlas          FastAPI ML engine (ai-engine/api.py, port 8000)
  (Patients, Caregivers,  ├─ POST /get_next_difficulty
   Reminders, GameSessions│     {reaction_time, mistakes, current_level}
   MemoryBankPhotos)      ├─ POST /calculate_health_score
                          ├─ POST /speak_regional_reminder
                          └─ POST /synthesize_speech  (Bhashini, optional)
  + Google Gemini (chat) + Meta WhatsApp Cloud API (reminders, optional)
```

Frontend routes: `/`, `/patient`, `/patient/login`, `/patient/reminders`, `/patient/games/*`, `/patient/family`, `/patient/profile`, `/caregiver`, `/caregiver/login`, `/caregiver/patient/:id`, `/caregiver/patient/:id/games`, `/caregiver/notifications`, `/caregiver/profile`, `/privacy-policy`.

## Environment variables

### Backend (`backend/.env`, see `backend/.env.example`)
| Variable | Required for demo? | Notes |
|---|---|---|
| `MONGO_URI` | Yes for full demo | MongoDB Atlas URI, or local `mongodb://localhost:27017/manomitra`. Without it the server starts but DB routes fail. No MongoDB is installed on this VM. |
| `JWT_SECRET` | Yes | Any long random string for demo. |
| `PORT` | No | Default 5000. Render sets it automatically. |
| `ML_SERVICE_URL` | No | Default is the original Render ML URL. For a local demo set `http://localhost:8000`. Timeout via `ML_SERVICE_TIMEOUT_MS` (default 6000). |
| `GEMINI_API_KEY` | No | Gemini chat only; without it the chat falls back. |
| `GOOGLE_CLIENT_ID` | No | Caregiver Google OAuth only. |
| `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`, `MY_WHATSAPP_NUMBER` | No | WhatsApp reminders only; demo works without them. |
| `BACKEND_URL`, `FRONTEND_URL`, `TRANSLATION_TIMEOUT_MS` | No | Used by cron/translation helpers. |

### Frontend (`frontend/.env`)
| Variable | Notes |
|---|---|
| `VITE_API_BASE_URL` or `VITE_API_URL` | Backend base URL, e.g. `http://localhost:5000`. The code appends `/api` if missing. Default without it is `http://localhost:5000/api`. |
| `VITE_GOOGLE_CLIENT_ID` | Caregiver Google login only. |

### AI engine (`ai-engine/.env`, see `ai-engine/.env.example`)
| Variable | Notes |
|---|---|
| `BHASHINI_USER_ID`, `BHASHINI_API_KEY`, `BHASHINI_INFERENCE_KEY` | Optional. Without them, regional reminders fall back to the small offline dictionary (Hindi/Bengali phrases only). |
| `PORT` | Default 8000 in the Dockerfile. |

## Run the demo (fastest, no Mongo, no keys)

This is the 1-day hackathon path — it proves the UI, games and ML, without WhatsApp/Gemini/Mongo:

```bash
# 1. ML engine (verified locally)
cd ai-engine
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pyttsx3
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
# test: POST /get_next_difficulty  {"reaction_time":2.5,"mistakes":0,"current_level":2} -> level 3
# test: POST /calculate_health_score {"patient_id":1,"games_played_this_week":3,"average_reaction_time":4.0,"total_mistakes_this_week":2} -> 93.5, Stable

# 2. Frontend only (mock fallback for game sessions; patient/caregiver login needs backend+DB)
cd frontend
npm install
npm run dev
# open the Vite URL, use the demo profiles shown on the login screen
```



Deploy notes: `render.yaml` deploys the Node backend (rootDir `backend`), `vercel.json` builds `frontend/` to `frontend/dist`. Set the same env vars in the Render/Vercel dashboards — never commit `.env` files.

## What is NOT verified yet
- Full backend + MongoDB run (no MongoDB on this VM; seed not run here).
- WhatsApp, Gemini, Google OAuth, biometric login, audio playback — all optional for the demo and untested here.
- Meera's caregiver detail record in the live site (only her roster card was seen in the earlier site scan).

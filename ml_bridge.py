#!/usr/bin/env python3
"""
ml_bridge.py — Manomitra ML Engine Bridge
---------------------------------------
Utility module that provides a clean Python interface to the
Manomitra Dementia AI Engine hosted on Render.

Endpoints exposed by the FastAPI ML microservice:
  POST /get_next_difficulty       → Adaptive game difficulty prediction
  POST /calculate_health_score    → Longitudinal cognitive health score (0-100)
  POST /speak_reminder            → Regional language TTS synthesis
  POST /synthesize_speech         → Bhashini speech pipeline

Usage (standalone test):
  python ml_bridge.py

Dependencies:
  pip install requests python-dotenv
"""

import os
import json
import time
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ML_ENGINE_BASE = os.getenv("ML_SERVICE_URL", "https://manomitra-ai-engine.onrender.com")
REQUEST_TIMEOUT = 12


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> Optional[dict]:
    url = f"{ML_ENGINE_BASE}{endpoint}"
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        elapsed = round((time.time() - start) * 1000)
        response.raise_for_status()
        print(f"[ML Bridge] {endpoint} → {response.status_code} ({elapsed}ms)")
        return response.json()
    except requests.exceptions.Timeout:
        print(f"[ML Bridge] ⚠️  Timeout — {endpoint}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[ML Bridge] ❌ Connection error — ML engine may be cold-starting.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[ML Bridge] ❌ HTTP {e.response.status_code} — {endpoint}")
        return None


def get_next_difficulty(patient_id, game_type, recent_accuracy, recent_reaction_time, recent_mistakes, current_level=1):
    payload = {
        "patient_id": patient_id, "game_type": game_type,
        "recent_accuracy": recent_accuracy, "recent_reaction_time": recent_reaction_time,
        "recent_mistakes": recent_mistakes, "current_level": current_level,
    }
    return _post("/get_next_difficulty", payload)


def calculate_health_score(patient_id, weekly_sessions):
    return _post("/calculate_health_score", {"patient_id": patient_id, "weekly_sessions": weekly_sessions})


def synthesize_speech(text, language="hi-IN"):
    return _post("/synthesize_speech", {"text": text, "language": language})


def ping_engine() -> bool:
    try:
        r = requests.get(f"{ML_ENGINE_BASE}/docs", timeout=REQUEST_TIMEOUT)
        alive = r.status_code == 200
        print(f"[ML Bridge] Engine ping → {'✅ reachable' if alive else f'⚠️ HTTP {r.status_code}'}")
        return alive
    except Exception as e:
        print(f"[ML Bridge] ❌ Unreachable: {e}")
        return False


# =============================================================================
# MANOMITRA CLINICAL REFERENCE DATASET
# Mock cognitive telemetry thresholds, normative benchmarks, and game metrics
# derived from published CST (Cognitive Stimulation Therapy) literature.
# Used for ML model calibration, rule-based fallback logic, and unit testing.
# =============================================================================

COGNITIVE_DOMAIN_THRESHOLDS = {
    "episodic_memory": {
        "domain": "Episodic & Working Memory",
        "neuroanatomy": "Hippocampus & Entorhinal Cortex",
        "game": "market-day-basket",
        "healthy_accuracy_min": 85,
        "mild_decline_accuracy_min": 65,
        "moderate_decline_accuracy_min": 45,
        "severe_decline_accuracy_max": 44,
        "healthy_reaction_max_s": 8.0,
        "mild_reaction_max_s": 14.0,
        "moderate_reaction_max_s": 22.0,
        "max_healthy_mistakes": 1,
        "max_mild_mistakes": 3,
        "max_moderate_mistakes": 6,
        "cst_evidence_level": "Grade A",
        "mmse_correlation": 0.74,
        "moca_correlation": 0.81,
    },
    "executive_function": {
        "domain": "Executive Function & Planning",
        "neuroanatomy": "Dorsolateral Prefrontal Cortex",
        "game": "daily-routine-sequencer",
        "healthy_accuracy_min": 80,
        "mild_decline_accuracy_min": 60,
        "moderate_decline_accuracy_min": 40,
        "severe_decline_accuracy_max": 39,
        "healthy_reaction_max_s": 10.0,
        "mild_reaction_max_s": 18.0,
        "moderate_reaction_max_s": 28.0,
        "max_healthy_mistakes": 1,
        "max_mild_mistakes": 3,
        "max_moderate_mistakes": 7,
        "cst_evidence_level": "Grade A",
        "mmse_correlation": 0.71,
        "moca_correlation": 0.78,
    },
    "facial_recognition": {
        "domain": "Facial Recognition & Reminiscence",
        "neuroanatomy": "Fusiform Gyrus (FFA) & Limbic System",
        "game": "faces-family-recall",
        "healthy_accuracy_min": 88,
        "mild_decline_accuracy_min": 68,
        "moderate_decline_accuracy_min": 48,
        "severe_decline_accuracy_max": 47,
        "healthy_reaction_max_s": 7.0,
        "mild_reaction_max_s": 12.0,
        "moderate_reaction_max_s": 20.0,
        "max_healthy_mistakes": 1,
        "max_mild_mistakes": 2,
        "max_moderate_mistakes": 5,
        "cst_evidence_level": "Grade B",
        "mmse_correlation": 0.68,
        "moca_correlation": 0.75,
    },
    "auditory_attention": {
        "domain": "Auditory Attention & Discrimination",
        "neuroanatomy": "Superior Temporal Gyrus & Auditory Cortex",
        "game": "sound-rhythm-match",
        "healthy_accuracy_min": 82,
        "mild_decline_accuracy_min": 62,
        "moderate_decline_accuracy_min": 42,
        "severe_decline_accuracy_max": 41,
        "healthy_reaction_max_s": 9.0,
        "mild_reaction_max_s": 16.0,
        "moderate_reaction_max_s": 25.0,
        "max_healthy_mistakes": 1,
        "max_mild_mistakes": 3,
        "max_moderate_mistakes": 6,
        "cst_evidence_level": "Grade B",
        "mmse_correlation": 0.65,
        "moca_correlation": 0.70,
    },
    "semantic_categorization": {
        "domain": "Semantic Categorization & Logic",
        "neuroanatomy": "Left Temporal Pole & Inferior Parietal Lobule",
        "game": "odd-one-out",
        "healthy_accuracy_min": 83,
        "mild_decline_accuracy_min": 63,
        "moderate_decline_accuracy_min": 43,
        "severe_decline_accuracy_max": 42,
        "healthy_reaction_max_s": 9.5,
        "mild_reaction_max_s": 17.0,
        "moderate_reaction_max_s": 26.0,
        "max_healthy_mistakes": 1,
        "max_mild_mistakes": 3,
        "max_moderate_mistakes": 6,
        "cst_evidence_level": "Grade A",
        "mmse_correlation": 0.72,
        "moca_correlation": 0.79,
    },
}

HEALTH_SCORE_BANDS = [
    {"min": 90, "max": 100, "label": "Excellent",        "color": "#22c55e", "action": "maintain"},
    {"min": 75, "max": 89,  "label": "Good",             "color": "#84cc16", "action": "maintain"},
    {"min": 60, "max": 74,  "label": "Stable",           "color": "#eab308", "action": "monitor"},
    {"min": 45, "max": 59,  "label": "Mild Decline",     "color": "#f97316", "action": "alert_caregiver"},
    {"min": 30, "max": 44,  "label": "Moderate Decline", "color": "#ef4444", "action": "urgent_alert"},
    {"min": 0,  "max": 29,  "label": "Severe Decline",   "color": "#991b1b", "action": "emergency"},
]

DIFFICULTY_LEVELS = {
    1: {"label": "Beginner",     "item_count": 3, "time_limit_s": 60, "hints": True,  "audio_cues": True},
    2: {"label": "Easy",         "item_count": 4, "time_limit_s": 45, "hints": True,  "audio_cues": True},
    3: {"label": "Intermediate", "item_count": 5, "time_limit_s": 35, "hints": False, "audio_cues": True},
    4: {"label": "Hard",         "item_count": 6, "time_limit_s": 25, "hints": False, "audio_cues": False},
    5: {"label": "Expert",       "item_count": 7, "time_limit_s": 18, "hints": False, "audio_cues": False},
}

NER_CULTURAL_ASSETS = {
    "produce": [
        {"name": "Kaji Nemu",        "name_as": "কাজি নেমু",     "category": "citrus",     "region": "Assam"},
        {"name": "Bhut Jolokia",     "name_as": "ভূত জলকীয়া",   "category": "spice",      "region": "Assam/Nagaland"},
        {"name": "Bamboo Shoot",     "name_as": "বাঁহ গাজ",      "category": "vegetable",  "region": "NER-wide"},
        {"name": "Assam Tea",        "name_as": "অসম চাহ",       "category": "beverage",   "region": "Assam"},
        {"name": "Malbhog Banana",   "name_as": "মালভোগ কল",     "category": "fruit",      "region": "Assam"},
        {"name": "Lakadong Turmeric","name_as": "লাকাডং হালধি",  "category": "spice",      "region": "Meghalaya"},
        {"name": "Naga King Chilli", "name_as": "নগা জলকীয়া",   "category": "spice",      "region": "Nagaland"},
        {"name": "Elephant Apple",   "name_as": "ওলক বগৰী",      "category": "fruit",      "region": "Assam"},
        {"name": "Bottle Gourd",     "name_as": "জিকা",          "category": "vegetable",  "region": "NER-wide"},
        {"name": "Black Sesame",     "name_as": "ক'লা তিল",      "category": "seed",       "region": "Assam"},
        {"name": "Water Hyacinth",   "name_as": "মেথনি",         "category": "vegetable",  "region": "Manipur"},
        {"name": "Roselle",          "name_as": "তেজপাত",        "category": "herb",       "region": "Assam"},
        {"name": "Wild Ginger",      "name_as": "আদা",           "category": "rhizome",    "region": "NER-wide"},
        {"name": "Star Anise",       "name_as": "তৰা মৌৰি",      "category": "spice",      "region": "Mizoram"},
        {"name": "Perilla",          "name_as": "সিচেয়",         "category": "herb",       "region": "Manipur"},
    ],
    "instruments": [
        {"name": "Dhol",       "name_as": "ঢোল",       "type": "percussion", "festival": "Bihu"},
        {"name": "Pepa",       "name_as": "পেপা",      "type": "wind",       "festival": "Bihu"},
        {"name": "Shankha",    "name_as": "শংখ",       "type": "wind",       "festival": "Puja"},
        {"name": "Dotara",     "name_as": "দোতৰা",     "type": "string",     "festival": "Bihu"},
        {"name": "Gogona",     "name_as": "গোগোনা",    "type": "idiophone",  "festival": "Bihu"},
        {"name": "Taal",       "name_as": "তাল",       "type": "percussion", "festival": "Puja"},
        {"name": "Kham",       "name_as": "খাম",       "type": "percussion", "festival": "Wangala"},
        {"name": "Duitara",    "name_as": "দুইতাৰা",   "type": "string",     "festival": "Folk"},
        {"name": "Bansuri",    "name_as": "বাঁহী",     "type": "wind",       "festival": "Bihu"},
        {"name": "Nagada",     "name_as": "নগাৰা",     "type": "percussion", "festival": "Assamese"},
    ],
    "festivals": [
        {"name": "Rongali Bihu",  "month": "April",     "state": "Assam",     "theme": "spring/harvest"},
        {"name": "Kongali Bihu",  "month": "October",   "state": "Assam",     "theme": "autumn/lamps"},
        {"name": "Bhogali Bihu",  "month": "January",   "state": "Assam",     "theme": "winter/feast"},
        {"name": "Wangala",       "month": "October",   "state": "Meghalaya", "theme": "post-harvest"},
        {"name": "Hornbill",      "month": "December",  "state": "Nagaland",  "theme": "cultural"},
        {"name": "Chapchar Kut",  "month": "March",     "state": "Mizoram",   "theme": "spring"},
        {"name": "Cheiraoba",     "month": "April",     "state": "Manipur",   "theme": "new-year"},
        {"name": "Mopin",         "month": "April",     "state": "Arunachal", "theme": "prosperity"},
        {"name": "Saga Dawa",     "month": "May/June",  "state": "Sikkim",    "theme": "Buddhist"},
        {"name": "Me-Dum-Me-Phi", "month": "January",   "state": "Assam",     "theme": "ancestor"},
        {"name": "Ali Aye Ligang","month": "February",  "state": "Assam",     "theme": "Mishing sowing"},
        {"name": "Dobur Uie",     "month": "October",   "state": "Assam",     "theme": "Deori"},
    ],
}

MOCK_PATIENT_TELEMETRY = [
    {"session_id": "s001", "patient_hash": "p_a1b2", "game": "market-day-basket",       "accuracy": 82.5, "reaction_time_s": 11.2, "mistakes": 1, "score": 78, "level": 3, "duration_s": 342, "week": 1, "day": 1},
    {"session_id": "s002", "patient_hash": "p_a1b2", "game": "odd-one-out",             "accuracy": 76.0, "reaction_time_s": 13.8, "mistakes": 2, "score": 71, "level": 2, "duration_s": 298, "week": 1, "day": 2},
    {"session_id": "s003", "patient_hash": "p_a1b2", "game": "sound-rhythm-match",      "accuracy": 79.0, "reaction_time_s": 12.5, "mistakes": 1, "score": 75, "level": 3, "duration_s": 315, "week": 1, "day": 3},
    {"session_id": "s004", "patient_hash": "p_a1b2", "game": "faces-family-recall",     "accuracy": 85.0, "reaction_time_s": 10.1, "mistakes": 0, "score": 88, "level": 3, "duration_s": 267, "week": 1, "day": 4},
    {"session_id": "s005", "patient_hash": "p_a1b2", "game": "daily-routine-sequencer", "accuracy": 73.0, "reaction_time_s": 15.3, "mistakes": 2, "score": 69, "level": 2, "duration_s": 410, "week": 1, "day": 5},
    {"session_id": "s006", "patient_hash": "p_a1b2", "game": "market-day-basket",       "accuracy": 80.0, "reaction_time_s": 11.9, "mistakes": 1, "score": 77, "level": 3, "duration_s": 330, "week": 1, "day": 6},
    {"session_id": "s007", "patient_hash": "p_a1b2", "game": "odd-one-out",             "accuracy": 78.5, "reaction_time_s": 12.8, "mistakes": 1, "score": 74, "level": 3, "duration_s": 305, "week": 1, "day": 7},
    {"session_id": "s008", "patient_hash": "p_c3d4", "game": "market-day-basket",       "accuracy": 91.0, "reaction_time_s":  8.4, "mistakes": 0, "score": 94, "level": 4, "duration_s": 220, "week": 1, "day": 1},
    {"session_id": "s009", "patient_hash": "p_c3d4", "game": "faces-family-recall",     "accuracy": 88.5, "reaction_time_s":  9.2, "mistakes": 0, "score": 90, "level": 4, "duration_s": 245, "week": 1, "day": 2},
    {"session_id": "s010", "patient_hash": "p_c3d4", "game": "daily-routine-sequencer", "accuracy": 86.0, "reaction_time_s": 10.0, "mistakes": 1, "score": 85, "level": 4, "duration_s": 280, "week": 1, "day": 3},
    {"session_id": "s011", "patient_hash": "p_c3d4", "game": "sound-rhythm-match",      "accuracy": 89.5, "reaction_time_s":  8.9, "mistakes": 0, "score": 91, "level": 4, "duration_s": 235, "week": 1, "day": 4},
    {"session_id": "s012", "patient_hash": "p_c3d4", "game": "odd-one-out",             "accuracy": 92.0, "reaction_time_s":  8.1, "mistakes": 0, "score": 95, "level": 5, "duration_s": 210, "week": 1, "day": 5},
    {"session_id": "s013", "patient_hash": "p_e5f6", "game": "market-day-basket",       "accuracy": 55.0, "reaction_time_s": 21.5, "mistakes": 4, "score": 48, "level": 1, "duration_s": 520, "week": 1, "day": 1},
    {"session_id": "s014", "patient_hash": "p_e5f6", "game": "odd-one-out",             "accuracy": 50.0, "reaction_time_s": 24.0, "mistakes": 5, "score": 42, "level": 1, "duration_s": 580, "week": 1, "day": 2},
    {"session_id": "s015", "patient_hash": "p_e5f6", "game": "sound-rhythm-match",      "accuracy": 48.0, "reaction_time_s": 26.3, "mistakes": 6, "score": 40, "level": 1, "duration_s": 610, "week": 1, "day": 3},
    {"session_id": "s016", "patient_hash": "p_e5f6", "game": "faces-family-recall",     "accuracy": 58.0, "reaction_time_s": 19.8, "mistakes": 3, "score": 52, "level": 1, "duration_s": 490, "week": 1, "day": 4},
    {"session_id": "s017", "patient_hash": "p_e5f6", "game": "daily-routine-sequencer", "accuracy": 52.0, "reaction_time_s": 23.1, "mistakes": 5, "score": 45, "level": 1, "duration_s": 560, "week": 1, "day": 5},
    {"session_id": "s018", "patient_hash": "p_g7h8", "game": "market-day-basket",       "accuracy": 68.0, "reaction_time_s": 16.2, "mistakes": 2, "score": 63, "level": 2, "duration_s": 385, "week": 2, "day": 1},
    {"session_id": "s019", "patient_hash": "p_g7h8", "game": "odd-one-out",             "accuracy": 71.5, "reaction_time_s": 15.0, "mistakes": 2, "score": 67, "level": 2, "duration_s": 360, "week": 2, "day": 2},
    {"session_id": "s020", "patient_hash": "p_g7h8", "game": "sound-rhythm-match",      "accuracy": 65.0, "reaction_time_s": 17.8, "mistakes": 3, "score": 60, "level": 2, "duration_s": 415, "week": 2, "day": 3},
    {"session_id": "s021", "patient_hash": "p_g7h8", "game": "faces-family-recall",     "accuracy": 74.0, "reaction_time_s": 14.5, "mistakes": 2, "score": 70, "level": 2, "duration_s": 345, "week": 2, "day": 4},
    {"session_id": "s022", "patient_hash": "p_g7h8", "game": "daily-routine-sequencer", "accuracy": 63.5, "reaction_time_s": 18.9, "mistakes": 3, "score": 58, "level": 2, "duration_s": 440, "week": 2, "day": 5},
    {"session_id": "s023", "patient_hash": "p_i9j0", "game": "market-day-basket",       "accuracy": 95.0, "reaction_time_s":  6.5, "mistakes": 0, "score": 98, "level": 5, "duration_s": 180, "week": 2, "day": 1},
    {"session_id": "s024", "patient_hash": "p_i9j0", "game": "odd-one-out",             "accuracy": 97.5, "reaction_time_s":  5.8, "mistakes": 0, "score": 100,"level": 5, "duration_s": 165, "week": 2, "day": 2},
    {"session_id": "s025", "patient_hash": "p_i9j0", "game": "sound-rhythm-match",      "accuracy": 93.0, "reaction_time_s":  7.1, "mistakes": 0, "score": 95, "level": 5, "duration_s": 195, "week": 2, "day": 3},
    {"session_id": "s026", "patient_hash": "p_i9j0", "game": "faces-family-recall",     "accuracy": 96.0, "reaction_time_s":  6.2, "mistakes": 0, "score": 99, "level": 5, "duration_s": 172, "week": 2, "day": 4},
    {"session_id": "s027", "patient_hash": "p_i9j0", "game": "daily-routine-sequencer", "accuracy": 94.5, "reaction_time_s":  6.9, "mistakes": 0, "score": 97, "level": 5, "duration_s": 188, "week": 2, "day": 5},
    {"session_id": "s028", "patient_hash": "p_k1l2", "game": "market-day-basket",       "accuracy": 61.0, "reaction_time_s": 19.5, "mistakes": 3, "score": 55, "level": 1, "duration_s": 465, "week": 3, "day": 1},
    {"session_id": "s029", "patient_hash": "p_k1l2", "game": "odd-one-out",             "accuracy": 58.0, "reaction_time_s": 22.0, "mistakes": 4, "score": 50, "level": 1, "duration_s": 510, "week": 3, "day": 2},
    {"session_id": "s030", "patient_hash": "p_k1l2", "game": "sound-rhythm-match",      "accuracy": 63.0, "reaction_time_s": 18.7, "mistakes": 3, "score": 57, "level": 1, "duration_s": 455, "week": 3, "day": 3},
    {"session_id": "s031", "patient_hash": "p_k1l2", "game": "faces-family-recall",     "accuracy": 70.0, "reaction_time_s": 16.4, "mistakes": 2, "score": 65, "level": 2, "duration_s": 380, "week": 3, "day": 4},
    {"session_id": "s032", "patient_hash": "p_k1l2", "game": "daily-routine-sequencer", "accuracy": 55.5, "reaction_time_s": 21.2, "mistakes": 4, "score": 48, "level": 1, "duration_s": 530, "week": 3, "day": 5},
    {"session_id": "s033", "patient_hash": "p_m3n4", "game": "market-day-basket",       "accuracy": 77.0, "reaction_time_s": 13.1, "mistakes": 2, "score": 73, "level": 3, "duration_s": 320, "week": 3, "day": 1},
    {"session_id": "s034", "patient_hash": "p_m3n4", "game": "odd-one-out",             "accuracy": 80.5, "reaction_time_s": 12.0, "mistakes": 1, "score": 78, "level": 3, "duration_s": 295, "week": 3, "day": 2},
    {"session_id": "s035", "patient_hash": "p_m3n4", "game": "sound-rhythm-match",      "accuracy": 75.0, "reaction_time_s": 14.2, "mistakes": 2, "score": 71, "level": 3, "duration_s": 340, "week": 3, "day": 3},
    {"session_id": "s036", "patient_hash": "p_m3n4", "game": "faces-family-recall",     "accuracy": 83.0, "reaction_time_s": 11.4, "mistakes": 1, "score": 81, "level": 3, "duration_s": 278, "week": 3, "day": 4},
    {"session_id": "s037", "patient_hash": "p_m3n4", "game": "daily-routine-sequencer", "accuracy": 72.5, "reaction_time_s": 15.8, "mistakes": 2, "score": 68, "level": 2, "duration_s": 395, "week": 3, "day": 5},
    {"session_id": "s038", "patient_hash": "p_o5p6", "game": "market-day-basket",       "accuracy": 88.0, "reaction_time_s":  9.6, "mistakes": 0, "score": 89, "level": 4, "duration_s": 238, "week": 4, "day": 1},
    {"session_id": "s039", "patient_hash": "p_o5p6", "game": "odd-one-out",             "accuracy": 85.5, "reaction_time_s": 10.3, "mistakes": 1, "score": 84, "level": 4, "duration_s": 255, "week": 4, "day": 2},
    {"session_id": "s040", "patient_hash": "p_o5p6", "game": "sound-rhythm-match",      "accuracy": 87.0, "reaction_time_s":  9.8, "mistakes": 0, "score": 87, "level": 4, "duration_s": 242, "week": 4, "day": 3},
    {"session_id": "s041", "patient_hash": "p_o5p6", "game": "faces-family-recall",     "accuracy": 90.5, "reaction_time_s":  8.7, "mistakes": 0, "score": 92, "level": 4, "duration_s": 225, "week": 4, "day": 4},
    {"session_id": "s042", "patient_hash": "p_o5p6", "game": "daily-routine-sequencer", "accuracy": 84.0, "reaction_time_s": 11.0, "mistakes": 1, "score": 82, "level": 4, "duration_s": 268, "week": 4, "day": 5},
    {"session_id": "s043", "patient_hash": "p_q7r8", "game": "market-day-basket",       "accuracy": 44.0, "reaction_time_s": 28.5, "mistakes": 7, "score": 35, "level": 1, "duration_s": 640, "week": 4, "day": 1},
    {"session_id": "s044", "patient_hash": "p_q7r8", "game": "odd-one-out",             "accuracy": 40.0, "reaction_time_s": 32.1, "mistakes": 8, "score": 30, "level": 1, "duration_s": 700, "week": 4, "day": 2},
    {"session_id": "s045", "patient_hash": "p_q7r8", "game": "sound-rhythm-match",      "accuracy": 38.5, "reaction_time_s": 34.0, "mistakes": 9, "score": 28, "level": 1, "duration_s": 740, "week": 4, "day": 3},
    {"session_id": "s046", "patient_hash": "p_q7r8", "game": "faces-family-recall",     "accuracy": 47.0, "reaction_time_s": 26.8, "mistakes": 6, "score": 38, "level": 1, "duration_s": 615, "week": 4, "day": 4},
    {"session_id": "s047", "patient_hash": "p_q7r8", "game": "daily-routine-sequencer", "accuracy": 42.5, "reaction_time_s": 30.4, "mistakes": 8, "score": 33, "level": 1, "duration_s": 675, "week": 4, "day": 5},
    {"session_id": "s048", "patient_hash": "p_s9t0", "game": "market-day-basket",       "accuracy": 72.0, "reaction_time_s": 14.8, "mistakes": 2, "score": 68, "level": 2, "duration_s": 365, "week": 5, "day": 1},
    {"session_id": "s049", "patient_hash": "p_s9t0", "game": "odd-one-out",             "accuracy": 75.5, "reaction_time_s": 13.6, "mistakes": 1, "score": 72, "level": 3, "duration_s": 338, "week": 5, "day": 2},
    {"session_id": "s050", "patient_hash": "p_s9t0", "game": "sound-rhythm-match",      "accuracy": 70.0, "reaction_time_s": 16.0, "mistakes": 2, "score": 65, "level": 2, "duration_s": 392, "week": 5, "day": 3},
    {"session_id": "s051", "patient_hash": "p_s9t0", "game": "faces-family-recall",     "accuracy": 78.0, "reaction_time_s": 12.9, "mistakes": 1, "score": 75, "level": 3, "duration_s": 312, "week": 5, "day": 4},
    {"session_id": "s052", "patient_hash": "p_s9t0", "game": "daily-routine-sequencer", "accuracy": 68.5, "reaction_time_s": 17.2, "mistakes": 2, "score": 63, "level": 2, "duration_s": 422, "week": 5, "day": 5},
    {"session_id": "s053", "patient_hash": "p_u1v2", "game": "market-day-basket",       "accuracy": 84.5, "reaction_time_s": 10.8, "mistakes": 1, "score": 83, "level": 4, "duration_s": 258, "week": 5, "day": 1},
    {"session_id": "s054", "patient_hash": "p_u1v2", "game": "odd-one-out",             "accuracy": 87.0, "reaction_time_s":  9.9, "mistakes": 0, "score": 86, "level": 4, "duration_s": 240, "week": 5, "day": 2},
    {"session_id": "s055", "patient_hash": "p_u1v2", "game": "sound-rhythm-match",      "accuracy": 82.5, "reaction_time_s": 11.5, "mistakes": 1, "score": 80, "level": 3, "duration_s": 285, "week": 5, "day": 3},
    {"session_id": "s056", "patient_hash": "p_u1v2", "game": "faces-family-recall",     "accuracy": 89.5, "reaction_time_s":  9.3, "mistakes": 0, "score": 90, "level": 4, "duration_s": 228, "week": 5, "day": 4},
    {"session_id": "s057", "patient_hash": "p_u1v2", "game": "daily-routine-sequencer", "accuracy": 81.0, "reaction_time_s": 12.2, "mistakes": 1, "score": 79, "level": 3, "duration_s": 300, "week": 5, "day": 5},
    {"session_id": "s058", "patient_hash": "p_w3x4", "game": "market-day-basket",       "accuracy": 66.5, "reaction_time_s": 17.5, "mistakes": 3, "score": 61, "level": 2, "duration_s": 420, "week": 6, "day": 1},
    {"session_id": "s059", "patient_hash": "p_w3x4", "game": "odd-one-out",             "accuracy": 62.0, "reaction_time_s": 19.8, "mistakes": 3, "score": 56, "level": 2, "duration_s": 468, "week": 6, "day": 2},
    {"session_id": "s060", "patient_hash": "p_w3x4", "game": "sound-rhythm-match",      "accuracy": 69.0, "reaction_time_s": 16.8, "mistakes": 2, "score": 64, "level": 2, "duration_s": 400, "week": 6, "day": 3},
    {"session_id": "s061", "patient_hash": "p_w3x4", "game": "faces-family-recall",     "accuracy": 73.5, "reaction_time_s": 15.1, "mistakes": 2, "score": 70, "level": 2, "duration_s": 358, "week": 6, "day": 4},
    {"session_id": "s062", "patient_hash": "p_w3x4", "game": "daily-routine-sequencer", "accuracy": 60.0, "reaction_time_s": 21.0, "mistakes": 4, "score": 53, "level": 1, "duration_s": 502, "week": 6, "day": 5},
    {"session_id": "s063", "patient_hash": "p_y5z6", "game": "market-day-basket",       "accuracy": 93.5, "reaction_time_s":  7.2, "mistakes": 0, "score": 96, "level": 5, "duration_s": 188, "week": 6, "day": 1},
    {"session_id": "s064", "patient_hash": "p_y5z6", "game": "odd-one-out",             "accuracy": 91.0, "reaction_time_s":  7.8, "mistakes": 0, "score": 93, "level": 5, "duration_s": 198, "week": 6, "day": 2},
    {"session_id": "s065", "patient_hash": "p_y5z6", "game": "sound-rhythm-match",      "accuracy": 94.5, "reaction_time_s":  6.8, "mistakes": 0, "score": 97, "level": 5, "duration_s": 178, "week": 6, "day": 3},
    {"session_id": "s066", "patient_hash": "p_y5z6", "game": "faces-family-recall",     "accuracy": 96.5, "reaction_time_s":  6.0, "mistakes": 0, "score": 99, "level": 5, "duration_s": 168, "week": 6, "day": 4},
    {"session_id": "s067", "patient_hash": "p_y5z6", "game": "daily-routine-sequencer", "accuracy": 90.0, "reaction_time_s":  8.5, "mistakes": 0, "score": 92, "level": 5, "duration_s": 215, "week": 6, "day": 5},
    {"session_id": "s068", "patient_hash": "p_aa12", "game": "market-day-basket",       "accuracy": 57.5, "reaction_time_s": 22.8, "mistakes": 4, "score": 50, "level": 1, "duration_s": 545, "week": 7, "day": 1},
    {"session_id": "s069", "patient_hash": "p_aa12", "game": "odd-one-out",             "accuracy": 54.0, "reaction_time_s": 25.3, "mistakes": 5, "score": 46, "level": 1, "duration_s": 590, "week": 7, "day": 2},
    {"session_id": "s070", "patient_hash": "p_aa12", "game": "sound-rhythm-match",      "accuracy": 60.0, "reaction_time_s": 20.7, "mistakes": 3, "score": 54, "level": 1, "duration_s": 498, "week": 7, "day": 3},
    {"session_id": "s071", "patient_hash": "p_bb34", "game": "market-day-basket",       "accuracy": 79.5, "reaction_time_s": 12.6, "mistakes": 1, "score": 76, "level": 3, "duration_s": 310, "week": 7, "day": 1},
    {"session_id": "s072", "patient_hash": "p_bb34", "game": "odd-one-out",             "accuracy": 83.0, "reaction_time_s": 11.3, "mistakes": 1, "score": 81, "level": 3, "duration_s": 282, "week": 7, "day": 2},
    {"session_id": "s073", "patient_hash": "p_bb34", "game": "sound-rhythm-match",      "accuracy": 76.5, "reaction_time_s": 13.9, "mistakes": 2, "score": 73, "level": 3, "duration_s": 332, "week": 7, "day": 3},
    {"session_id": "s074", "patient_hash": "p_cc56", "game": "market-day-basket",       "accuracy": 86.5, "reaction_time_s": 10.5, "mistakes": 0, "score": 85, "level": 4, "duration_s": 252, "week": 8, "day": 1},
    {"session_id": "s075", "patient_hash": "p_cc56", "game": "odd-one-out",             "accuracy": 84.0, "reaction_time_s": 11.1, "mistakes": 1, "score": 82, "level": 4, "duration_s": 270, "week": 8, "day": 2},
    {"session_id": "s076", "patient_hash": "p_cc56", "game": "sound-rhythm-match",      "accuracy": 88.5, "reaction_time_s":  9.7, "mistakes": 0, "score": 88, "level": 4, "duration_s": 235, "week": 8, "day": 3},
    {"session_id": "s077", "patient_hash": "p_cc56", "game": "faces-family-recall",     "accuracy": 91.0, "reaction_time_s":  8.6, "mistakes": 0, "score": 92, "level": 4, "duration_s": 222, "week": 8, "day": 4},
    {"session_id": "s078", "patient_hash": "p_cc56", "game": "daily-routine-sequencer", "accuracy": 85.5, "reaction_time_s": 10.8, "mistakes": 1, "score": 84, "level": 4, "duration_s": 260, "week": 8, "day": 5},
    {"session_id": "s079", "patient_hash": "p_dd78", "game": "market-day-basket",       "accuracy": 46.0, "reaction_time_s": 27.2, "mistakes": 7, "score": 37, "level": 1, "duration_s": 628, "week": 8, "day": 1},
    {"session_id": "s080", "patient_hash": "p_dd78", "game": "odd-one-out",             "accuracy": 43.5, "reaction_time_s": 30.5, "mistakes": 8, "score": 33, "level": 1, "duration_s": 685, "week": 8, "day": 2},
    {"session_id": "s081", "patient_hash": "p_ee90", "game": "market-day-basket",       "accuracy": 76.0, "reaction_time_s": 13.5, "mistakes": 2, "score": 72, "level": 3, "duration_s": 325, "week": 9, "day": 1},
    {"session_id": "s082", "patient_hash": "p_ee90", "game": "odd-one-out",             "accuracy": 79.0, "reaction_time_s": 12.3, "mistakes": 1, "score": 76, "level": 3, "duration_s": 302, "week": 9, "day": 2},
    {"session_id": "s083", "patient_hash": "p_ee90", "game": "sound-rhythm-match",      "accuracy": 74.5, "reaction_time_s": 14.8, "mistakes": 2, "score": 70, "level": 2, "duration_s": 352, "week": 9, "day": 3},
    {"session_id": "s084", "patient_hash": "p_ee90", "game": "faces-family-recall",     "accuracy": 81.0, "reaction_time_s": 12.0, "mistakes": 1, "score": 79, "level": 3, "duration_s": 290, "week": 9, "day": 4},
    {"session_id": "s085", "patient_hash": "p_ee90", "game": "daily-routine-sequencer", "accuracy": 71.0, "reaction_time_s": 16.5, "mistakes": 2, "score": 66, "level": 2, "duration_s": 405, "week": 9, "day": 5},
    {"session_id": "s086", "patient_hash": "p_ff12", "game": "market-day-basket",       "accuracy": 92.5, "reaction_time_s":  7.8, "mistakes": 0, "score": 94, "level": 5, "duration_s": 195, "week": 9, "day": 1},
    {"session_id": "s087", "patient_hash": "p_ff12", "game": "odd-one-out",             "accuracy": 94.0, "reaction_time_s":  7.2, "mistakes": 0, "score": 96, "level": 5, "duration_s": 182, "week": 9, "day": 2},
    {"session_id": "s088", "patient_hash": "p_ff12", "game": "sound-rhythm-match",      "accuracy": 91.5, "reaction_time_s":  8.0, "mistakes": 0, "score": 93, "level": 5, "duration_s": 202, "week": 9, "day": 3},
    {"session_id": "s089", "patient_hash": "p_gg34", "game": "market-day-basket",       "accuracy": 64.0, "reaction_time_s": 18.2, "mistakes": 3, "score": 59, "level": 2, "duration_s": 435, "week": 10, "day": 1},
    {"session_id": "s090", "patient_hash": "p_gg34", "game": "odd-one-out",             "accuracy": 67.5, "reaction_time_s": 16.9, "mistakes": 2, "score": 63, "level": 2, "duration_s": 408, "week": 10, "day": 2},
    {"session_id": "s091", "patient_hash": "p_gg34", "game": "sound-rhythm-match",      "accuracy": 61.5, "reaction_time_s": 20.1, "mistakes": 3, "score": 56, "level": 2, "duration_s": 475, "week": 10, "day": 3},
    {"session_id": "s092", "patient_hash": "p_gg34", "game": "faces-family-recall",     "accuracy": 70.5, "reaction_time_s": 16.0, "mistakes": 2, "score": 66, "level": 2, "duration_s": 382, "week": 10, "day": 4},
    {"session_id": "s093", "patient_hash": "p_gg34", "game": "daily-routine-sequencer", "accuracy": 59.0, "reaction_time_s": 21.8, "mistakes": 4, "score": 52, "level": 1, "duration_s": 520, "week": 10, "day": 5},
    {"session_id": "s094", "patient_hash": "p_hh56", "game": "market-day-basket",       "accuracy": 85.0, "reaction_time_s": 10.9, "mistakes": 1, "score": 84, "level": 4, "duration_s": 262, "week": 10, "day": 1},
    {"session_id": "s095", "patient_hash": "p_hh56", "game": "odd-one-out",             "accuracy": 88.0, "reaction_time_s":  9.8, "mistakes": 0, "score": 87, "level": 4, "duration_s": 242, "week": 10, "day": 2},
    {"session_id": "s096", "patient_hash": "p_hh56", "game": "sound-rhythm-match",      "accuracy": 83.5, "reaction_time_s": 11.2, "mistakes": 1, "score": 82, "level": 3, "duration_s": 278, "week": 10, "day": 3},
    {"session_id": "s097", "patient_hash": "p_hh56", "game": "faces-family-recall",     "accuracy": 90.0, "reaction_time_s":  9.0, "mistakes": 0, "score": 90, "level": 4, "duration_s": 228, "week": 10, "day": 4},
    {"session_id": "s098", "patient_hash": "p_hh56", "game": "daily-routine-sequencer", "accuracy": 82.0, "reaction_time_s": 12.0, "mistakes": 1, "score": 80, "level": 3, "duration_s": 295, "week": 10, "day": 5},
    {"session_id": "s099", "patient_hash": "p_ii78", "game": "market-day-basket",       "accuracy": 53.0, "reaction_time_s": 24.6, "mistakes": 5, "score": 45, "level": 1, "duration_s": 572, "week": 11, "day": 1},
    {"session_id": "s100", "patient_hash": "p_ii78", "game": "odd-one-out",             "accuracy": 49.5, "reaction_time_s": 27.3, "mistakes": 6, "score": 41, "level": 1, "duration_s": 625, "week": 11, "day": 2},
]

BHASHINI_LANGUAGE_CODES = {
    "as":  {"bcp47": "as-IN",  "name": "Assamese",  "script": "Bengali",    "tts_supported": True,  "asr_supported": True},
    "bn":  {"bcp47": "bn-IN",  "name": "Bengali",   "script": "Bengali",    "tts_supported": True,  "asr_supported": True},
    "bo":  {"bcp47": "bo-IN",  "name": "Bodo",      "script": "Devanagari", "tts_supported": False, "asr_supported": False},
    "en":  {"bcp47": "en-IN",  "name": "English",   "script": "Latin",      "tts_supported": True,  "asr_supported": True},
    "gu":  {"bcp47": "gu-IN",  "name": "Gujarati",  "script": "Gujarati",   "tts_supported": True,  "asr_supported": True},
    "hi":  {"bcp47": "hi-IN",  "name": "Hindi",     "script": "Devanagari", "tts_supported": True,  "asr_supported": True},
    "kn":  {"bcp47": "kn-IN",  "name": "Kannada",   "script": "Kannada",    "tts_supported": True,  "asr_supported": True},
    "ks":  {"bcp47": "ks-IN",  "name": "Kashmiri",  "script": "Perso-Arab", "tts_supported": False, "asr_supported": False},
    "mai": {"bcp47": "mai-IN", "name": "Maithili",  "script": "Devanagari", "tts_supported": False, "asr_supported": False},
    "ml":  {"bcp47": "ml-IN",  "name": "Malayalam", "script": "Malayalam",  "tts_supported": True,  "asr_supported": True},
    "mni": {"bcp47": "mni-IN", "name": "Meitei",    "script": "Meitei",     "tts_supported": False, "asr_supported": False},
    "mr":  {"bcp47": "mr-IN",  "name": "Marathi",   "script": "Devanagari", "tts_supported": True,  "asr_supported": True},
    "ne":  {"bcp47": "ne-IN",  "name": "Nepali",    "script": "Devanagari", "tts_supported": True,  "asr_supported": False},
    "or":  {"bcp47": "or-IN",  "name": "Odia",      "script": "Odia",       "tts_supported": True,  "asr_supported": True},
    "pa":  {"bcp47": "pa-IN",  "name": "Punjabi",   "script": "Gurmukhi",   "tts_supported": True,  "asr_supported": True},
    "sa":  {"bcp47": "sa-IN",  "name": "Sanskrit",  "script": "Devanagari", "tts_supported": False, "asr_supported": False},
    "sat": {"bcp47": "sat-IN", "name": "Santali",   "script": "Ol Chiki",   "tts_supported": False, "asr_supported": False},
    "sd":  {"bcp47": "sd-IN",  "name": "Sindhi",    "script": "Devanagari", "tts_supported": False, "asr_supported": False},
    "ta":  {"bcp47": "ta-IN",  "name": "Tamil",     "script": "Tamil",      "tts_supported": True,  "asr_supported": True},
    "te":  {"bcp47": "te-IN",  "name": "Telugu",    "script": "Telugu",     "tts_supported": True,  "asr_supported": True},
    "ur":  {"bcp47": "ur-IN",  "name": "Urdu",      "script": "Perso-Arab", "tts_supported": True,  "asr_supported": True},
}

NER_CLINICAL_BENCHMARKS = {
    "assam": {
        "state": "Assam", "population_60_plus": 3_200_000,
        "estimated_dementia_cases": 180_000, "prevalence_rate_pct": 5.6,
        "neurologists_total": 42, "neurologists_outside_guwahati": 6,
        "hypertension_prevalence_pct": 33.2, "vascular_dementia_share_pct": 28.5,
        "primary_language": "Assamese", "secondary_languages": ["Bodo", "Bengali", "Mising"],
        "ardsi_chapter": True, "tele_manas_nodes": 4,
    },
    "meghalaya": {
        "state": "Meghalaya", "population_60_plus": 310_000,
        "estimated_dementia_cases": 22_000, "prevalence_rate_pct": 7.1,
        "neurologists_total": 4, "neurologists_outside_shillong": 0,
        "hypertension_prevalence_pct": 29.8, "vascular_dementia_share_pct": 31.2,
        "primary_language": "Khasi", "secondary_languages": ["Garo", "English", "Bengali"],
        "ardsi_chapter": False, "tele_manas_nodes": 1,
    },
    "nagaland": {
        "state": "Nagaland", "population_60_plus": 240_000,
        "estimated_dementia_cases": 16_500, "prevalence_rate_pct": 6.9,
        "neurologists_total": 3, "neurologists_outside_kohima": 0,
        "hypertension_prevalence_pct": 27.5, "vascular_dementia_share_pct": 26.8,
        "primary_language": "Nagamese", "secondary_languages": ["English", "Hindi"],
        "ardsi_chapter": False, "tele_manas_nodes": 1,
    },
    "manipur": {
        "state": "Manipur", "population_60_plus": 290_000,
        "estimated_dementia_cases": 19_800, "prevalence_rate_pct": 6.8,
        "neurologists_total": 5, "neurologists_outside_imphal": 1,
        "hypertension_prevalence_pct": 31.4, "vascular_dementia_share_pct": 29.9,
        "primary_language": "Meitei", "secondary_languages": ["English", "Hindi"],
        "ardsi_chapter": False, "tele_manas_nodes": 2,
    },
    "mizoram": {
        "state": "Mizoram", "population_60_plus": 110_000,
        "estimated_dementia_cases": 8_200, "prevalence_rate_pct": 7.5,
        "neurologists_total": 2, "neurologists_outside_aizawl": 0,
        "hypertension_prevalence_pct": 35.1, "vascular_dementia_share_pct": 33.4,
        "primary_language": "Mizo", "secondary_languages": ["English"],
        "ardsi_chapter": True, "tele_manas_nodes": 1,
    },
    "tripura": {
        "state": "Tripura", "population_60_plus": 420_000,
        "estimated_dementia_cases": 28_000, "prevalence_rate_pct": 6.7,
        "neurologists_total": 7, "neurologists_outside_agartala": 1,
        "hypertension_prevalence_pct": 28.9, "vascular_dementia_share_pct": 27.1,
        "primary_language": "Bengali", "secondary_languages": ["Kokborok", "Hindi"],
        "ardsi_chapter": False, "tele_manas_nodes": 2,
    },
    "arunachal": {
        "state": "Arunachal Pradesh", "population_60_plus": 160_000,
        "estimated_dementia_cases": 10_500, "prevalence_rate_pct": 6.6,
        "neurologists_total": 2, "neurologists_outside_itanagar": 0,
        "hypertension_prevalence_pct": 26.3, "vascular_dementia_share_pct": 25.5,
        "primary_language": "Nyishi", "secondary_languages": ["English", "Hindi", "Adi"],
        "ardsi_chapter": False, "tele_manas_nodes": 1,
    },
    "sikkim": {
        "state": "Sikkim", "population_60_plus": 80_000,
        "estimated_dementia_cases": 5_200, "prevalence_rate_pct": 6.5,
        "neurologists_total": 2, "neurologists_outside_gangtok": 0,
        "hypertension_prevalence_pct": 24.8, "vascular_dementia_share_pct": 24.2,
        "primary_language": "Nepali", "secondary_languages": ["Sikkimese", "English", "Hindi"],
        "ardsi_chapter": False, "tele_manas_nodes": 1,
    },
}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Manomitra ML Engine Bridge — Connectivity Test")
    print(f"Target: {ML_ENGINE_BASE}")
    print(f"Mock telemetry records loaded: {len(MOCK_PATIENT_TELEMETRY)}")
    print(f"NER states benchmarked: {len(NER_CLINICAL_BENCHMARKS)}")
    print(f"Cultural assets — produce: {len(NER_CULTURAL_ASSETS['produce'])}, "
          f"instruments: {len(NER_CULTURAL_ASSETS['instruments'])}, "
          f"festivals: {len(NER_CULTURAL_ASSETS['festivals'])}")
    print("=" * 60)

    alive = ping_engine()
    if not alive:
        print("\n⚠️  Engine cold-starting (Render free tier). Wait ~30s and retry.\n")

    print("\n[Test] get_next_difficulty")
    result = get_next_difficulty(
        patient_id=1, game_type="market-day-basket",
        recent_accuracy=78.5, recent_reaction_time=12.3,
        recent_mistakes=1, current_level=2,
    )
    if result:
        print(json.dumps(result, indent=2))

    print("\n[ML Bridge] Test complete.")

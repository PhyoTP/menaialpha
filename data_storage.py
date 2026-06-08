import json
import os
from datetime import datetime

HISTORY_FILE = "chat_sessions.json"


def load_all_sessions():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_message(session_id, role, content):
    sessions = load_all_sessions()

    # Find existing session or create new one
    session = next((s for s in sessions if s["session_id"] == session_id), None)
    if not session:
        session = {"session_id": session_id, "messages": [], "created_at": datetime.now().isoformat()}
        sessions.append(session)

    session["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(sessions, f, indent=2)
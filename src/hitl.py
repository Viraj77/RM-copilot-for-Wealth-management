import json
from datetime import datetime
import uuid
from pathlib import Path

LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "human_review_log.json"


# ====================================================
# CREATE REVIEW TICKET
# ====================================================

def create_review_ticket(query):

    ticket = {

        "ticket_id": f"TICKET-{uuid.uuid4().hex[:8].upper()}",

        "query": query,

        "status": "Pending Human Review",

        "priority": "High",

        "created_at": str(datetime.now())
    }

    return ticket



# ====================================================
# LOG ESCALATION
# ====================================================

def log_escalation(ticket):

    with open(LOG_FILE, "a") as f:

        f.write(
            json.dumps(ticket)
        )

        f.write("\n")
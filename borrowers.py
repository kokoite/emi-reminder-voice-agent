"""Mock borrowers. Due dates are stored relative to today, so the data never goes stale."""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass(frozen=True)
class Borrower:
    borrower_id: str
    name: str
    birth_date: date
    last4: str  # last four digits of the registered mobile number
    loan_type: str
    emi_amount: int  # rupees
    due_date: date
    days_overdue: int  # positive: overdue; zero: due today; negative: due in that many days
    preferred_language: str
    scenario: str


def load_borrowers(path: Path, today: date) -> dict[str, Borrower]:
    """Read the mock borrowers, keyed by borrower_id, with due dates set from `today`."""
    rows = json.loads(path.read_text())
    borrowers = {}
    for row in rows:
        due_in_days = row["due_in_days"]
        borrowers[row["borrower_id"]] = Borrower(
            borrower_id=row["borrower_id"],
            name=row["name"],
            birth_date=date.fromisoformat(row["birth_date"]),
            last4=row["last4"],
            loan_type=row["loan_type"],
            emi_amount=row["emi_amount"],
            due_date=today + timedelta(days=due_in_days),
            days_overdue=-due_in_days,
            preferred_language=row["preferred_language"],
            scenario=row["scenario"],
        )
    return borrowers

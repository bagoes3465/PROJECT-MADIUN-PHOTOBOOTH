"""
Mood statistics router
Provides aggregated face-expression stats for the homepage
("Mood Kota Madiun minggu ini").
"""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter
from database import get_supabase

router = APIRouter(prefix="/photobooth", tags=["mood"])

# Must match the 3-class model: happy, normal, sad
EXPRESSION_LABELS = {
    "happy": "Senang",
    "normal": "Netral",
    "sad": "Sedih",
}


@router.get("/mood/weekly")
async def get_weekly_mood():
    """
    Aggregate face_expressions from the last rolling 7 days.

    Returns the count and percentage of each expression, plus the
    dominant (most frequent) one — for display on the homepage.
    """
    db = get_supabase()

    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    result = (
        db.table("face_expressions")
        .select("expression")
        .gte("created_at", since)
        .execute()
    )

    rows = result.data or []
    total = len(rows)

    # Initialize all known expressions at 0 so the UI always has all 3 keys,
    # even if one of them had zero occurrences this week.
    counts = {expr: 0 for expr in EXPRESSION_LABELS}
    for row in rows:
        expr = row.get("expression")
        if expr in counts:
            counts[expr] += 1
        # Unknown/legacy expression values are ignored in the aggregate
        # rather than crashing the endpoint.

    breakdown = []
    for expr, label in EXPRESSION_LABELS.items():
        count = counts[expr]
        percentage = round((count / total) * 100, 1) if total > 0 else 0.0
        breakdown.append({
            "expression": expr,
            "expression_label": label,
            "count": count,
            "percentage": percentage,
        })

    dominant = max(breakdown, key=lambda b: b["count"]) if total > 0 else None

    return {
        "success": True,
        "message": "Weekly mood retrieved",
        "data": {
            "period_days": 7,
            "since": since,
            "total_samples": total,
            "dominant_expression": dominant["expression"] if dominant else None,
            "dominant_expression_label": dominant["expression_label"] if dominant else None,
            "breakdown": breakdown,
        },
    }       
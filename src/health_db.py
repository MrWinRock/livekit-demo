"""Health records database backed by SQLite.

Provides read-only query functions for the agent. Call init_db() at startup.
The schema stores one row per health check; the agent always fetches the
most-recent row per user via get_latest_health().
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path

HEALTH_DB_PATH = Path(__file__).parent.parent / "data" / "health.db"

_SEED_RECORDS: list[tuple] = [
    # (uuid, name, gender, age_years, age_months, age_days, user_id, updated_at,
    #  height_cm, weight_kg, bmr, bmi, hbpm, blood_pressure, o2, body_temp)
    (
        "a1b2c3d4-0001-0001-0001-000000000001",
        "ธีรวัฒน์ มั่นคง",
        "male",
        35,
        2,
        15,
        "user_001",
        "2026-04-28 09:15:00",
        175.0,
        85.0,
        1868.0,
        27.8,
        88,
        "128/84",
        97.0,
        36.8,
    ),
    (
        "a1b2c3d4-0001-0001-0001-000000000002",
        "ธีรวัฒน์ มั่นคง",
        "male",
        35,
        3,
        0,
        "user_001",
        "2026-05-01 08:30:00",
        175.0,
        86.5,
        1886.0,
        28.2,
        92,
        "132/86",
        97.0,
        36.9,
    ),
    (
        "a1b2c3d4-0002-0002-0002-000000000001",
        "สุนิสา ใจดี",
        "female",
        28,
        7,
        10,
        "user_002",
        "2026-05-02 10:00:00",
        162.0,
        55.0,
        1337.0,
        20.9,
        72,
        "112/74",
        93.5,
        36.5,
    ),
    (
        "a1b2c3d4-0003-0003-0003-000000000001",
        "สมชาย วงศ์ใหญ่",
        "male",
        50,
        1,
        5,
        "user_003",
        "2026-04-30 07:45:00",
        168.0,
        95.0,
        1883.0,
        33.7,
        105,
        "145/92",
        96.0,
        37.1,
    ),
    (
        "a1b2c3d4-0004-0004-0004-000000000001",
        "พิมพ์ใจ สุขสวัสดิ์",
        "female",
        42,
        0,
        20,
        "user_004",
        "2026-05-04 14:20:00",
        158.0,
        57.0,
        1282.0,
        22.8,
        68,
        "116/76",
        98.0,
        36.6,
    ),
]


@dataclass(frozen=True)
class HealthRecord:
    uuid: str
    name: str
    gender: str
    age_years: int
    age_months: int
    age_days: int
    user_id: str
    updated_at: str
    height_cm: float
    weight_kg: float
    bmr: float
    bmi: float
    hbpm: int
    blood_pressure: str
    o2: float
    body_temp: float

    def to_summary(self) -> str:
        return (
            f"Name: {self.name} | Gender: {self.gender} | "
            f"Age: {self.age_years}y {self.age_months}m {self.age_days}d | "
            f"Height: {self.height_cm} cm | Weight: {self.weight_kg} kg | "
            f"BMI: {self.bmi:.1f} | BMR: {self.bmr:.0f} kcal/day | "
            f"Heart rate: {self.hbpm} bpm | Blood pressure: {self.blood_pressure} mmHg | "
            f"O2 saturation: {self.o2}% | Body temp: {self.body_temp}°C | "
            f"Updated: {self.updated_at}"
        )

    def range_analysis(self) -> str:
        """Classify each metric against standard reference ranges."""
        try:
            sys_bp, dia_bp = (int(x) for x in self.blood_pressure.split("/"))
        except ValueError:
            sys_bp, dia_bp = 0, 0

        def _bmi_label(v: float) -> str:
            if v < 18.5:
                return f"UNDERWEIGHT (< 18.5) — current {v:.1f}"
            if v <= 24.9:
                return f"NORMAL (18.5-24.9) — current {v:.1f}"
            if v <= 29.9:
                return f"OVERWEIGHT (25-29.9) — current {v:.1f}"
            return f"OBESE (≥ 30) — current {v:.1f}"

        def _hr_label(v: int) -> str:
            if v < 60:
                return f"LOW/BRADYCARDIA (< 60) — current {v}"
            if v <= 100:
                return f"NORMAL (60-100) — current {v}"
            return f"HIGH/TACHYCARDIA (> 100) — current {v}"

        def _bp_label(s: int, d: int) -> str:
            if s < 120 and d < 80:
                return f"NORMAL (< 120/80) — current {s}/{d}"
            if s < 130 and d < 80:
                return f"ELEVATED (120-129/<80) — current {s}/{d}"
            if s <= 139 or d <= 89:
                return f"HIGH STAGE 1 (130-139/80-89) — current {s}/{d}"
            return f"HIGH STAGE 2 (≥ 140/≥ 90) — current {s}/{d}"

        def _o2_label(v: float) -> str:
            if v >= 95:
                return f"NORMAL (≥ 95%) — current {v}"
            if v >= 90:
                return f"LOW (90-94%) — current {v}"
            return f"CRITICALLY LOW (< 90%) — current {v}"

        def _temp_label(v: float) -> str:
            if v < 36.1:
                return f"LOW/HYPOTHERMIA (< 36.1°C) — current {v}"
            if v <= 37.2:
                return f"NORMAL (36.1-37.2°C) — current {v}"
            if v <= 38.0:
                return f"SLIGHTLY ELEVATED (37.3-38.0°C) — current {v}"
            return f"FEVER (> 38.0°C) — current {v}"

        lines = [
            f"BMI: {_bmi_label(self.bmi)}",
            f"Heart rate: {_hr_label(self.hbpm)}",
            f"Blood pressure: {_bp_label(sys_bp, dia_bp)}",
            f"O2 saturation: {_o2_label(self.o2)}",
            f"Body temperature: {_temp_label(self.body_temp)}",
        ]
        return "\n".join(lines)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = HEALTH_DB_PATH) -> None:
    """Create the health table and seed sample data if empty. Safe to call repeatedly."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS health (
                uuid           TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                gender         TEXT NOT NULL,
                age_years      INTEGER NOT NULL,
                age_months     INTEGER NOT NULL DEFAULT 0,
                age_days       INTEGER NOT NULL DEFAULT 0,
                user_id        TEXT NOT NULL,
                updated_at     TEXT NOT NULL,
                height_cm      REAL NOT NULL,
                weight_kg      REAL NOT NULL,
                bmr            REAL NOT NULL,
                bmi            REAL NOT NULL,
                hbpm           INTEGER NOT NULL,
                blood_pressure TEXT NOT NULL,
                o2             REAL NOT NULL,
                body_temp      REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_health_user_updated ON health (user_id, updated_at DESC)"
        )
        if conn.execute("SELECT COUNT(*) FROM health").fetchone()[0] == 0:
            conn.executemany(
                """
                INSERT INTO health
                    (uuid, name, gender, age_years, age_months, age_days,
                     user_id, updated_at, height_cm, weight_kg, bmr, bmi,
                     hbpm, blood_pressure, o2, body_temp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                _SEED_RECORDS,
            )
        conn.commit()


def _row_to_record(row: sqlite3.Row) -> HealthRecord:
    return HealthRecord(
        uuid=row["uuid"],
        name=row["name"],
        gender=row["gender"],
        age_years=row["age_years"],
        age_months=row["age_months"],
        age_days=row["age_days"],
        user_id=row["user_id"],
        updated_at=row["updated_at"],
        height_cm=row["height_cm"],
        weight_kg=row["weight_kg"],
        bmr=row["bmr"],
        bmi=row["bmi"],
        hbpm=row["hbpm"],
        blood_pressure=row["blood_pressure"],
        o2=row["o2"],
        body_temp=row["body_temp"],
    )


def get_latest_health(
    user_id: str, db_path: str | Path = HEALTH_DB_PATH
) -> HealthRecord | None:
    """Return the most-recent health record for the given user_id."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM health WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return _row_to_record(row) if row else None


def get_latest_health_by_name(
    name: str, db_path: str | Path = HEALTH_DB_PATH
) -> HealthRecord | None:
    """Return the most-recent health record matching the full name (case-insensitive)."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM health WHERE name LIKE ? COLLATE NOCASE ORDER BY updated_at DESC LIMIT 1",
            (f"%{name.strip()}%",),
        ).fetchone()
    return _row_to_record(row) if row else None


def get_most_recent_health(db_path: str | Path = HEALTH_DB_PATH) -> HealthRecord | None:
    """Return the single most-recently-updated record across all users."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM health ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    return _row_to_record(row) if row else None


def list_users(db_path: str | Path = HEALTH_DB_PATH) -> list[dict]:
    """Return distinct users with their most-recent update timestamp."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT user_id, name, MAX(updated_at) AS last_updated
            FROM health
            GROUP BY user_id
            ORDER BY last_updated DESC
            """,
        ).fetchall()
    return [
        {"user_id": r["user_id"], "name": r["name"], "last_updated": r["last_updated"]}
        for r in rows
    ]


async def get_latest_health_async(
    user_id: str | None = None,
    name: str | None = None,
    db_path: str | Path = HEALTH_DB_PATH,
) -> HealthRecord | None:
    """Async wrapper — looks up by user_id first, then by name."""

    def _query() -> HealthRecord | None:
        if user_id:
            rec = get_latest_health(user_id, db_path)
            if rec:
                return rec
        if name:
            return get_latest_health_by_name(name, db_path)
        return None

    return await asyncio.to_thread(_query)

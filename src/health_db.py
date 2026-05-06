"""Health records database backed by SQLite.

Provides read-only query functions for the agent. Call init_db() at startup.
The schema stores one row per health check; the agent always fetches the
most-recent row per user via get_latest_health().
"""

from __future__ import annotations

import asyncio
import calendar
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

HEALTH_DB_PATH = Path(__file__).parent.parent / "data" / "health.db"

_SEED_RECORDS: list[tuple] = [
    # (uuid, name, gender, date_of_birth, user_id, updated_at, height_cm,
    #  weight_kg, hbpm, blood_pressure, o2, body_temp)
    (
        "a1b2c3d4-0001-0001-0001-000000000001",
        "ธีรวัฒน์ มั่นคง",
        "male",
        "1990-01-13 00:00:00",
        "user_001",
        "2026-04-28 09:15:00",
        175.0,
        85.0,
        88,
        "128/84",
        97.0,
        36.8,
    ),
    (
        "a1b2c3d4-0001-0001-0001-000000000002",
        "ธีรวัฒน์ มั่นคง",
        "male",
        "1990-01-31 00:00:00",
        "user_001",
        "2026-05-01 08:30:00",
        175.0,
        86.5,
        92,
        "132/86",
        97.0,
        36.9,
    ),
    (
        "a1b2c3d4-0002-0002-0002-000000000001",
        "สุนิสา ใจดี",
        "female",
        "1996-09-22 00:00:00",
        "user_002",
        "2026-05-02 10:00:00",
        162.0,
        55.0,
        72,
        "112/74",
        93.5,
        36.5,
    ),
    (
        "a1b2c3d4-0003-0003-0003-000000000001",
        "สมชาย วงศ์ใหญ่",
        "male",
        "1976-03-25 00:00:00",
        "user_003",
        "2026-04-30 07:45:00",
        168.0,
        95.0,
        105,
        "145/92",
        96.0,
        37.1,
    ),
    (
        "a1b2c3d4-0004-0004-0004-000000000001",
        "พิมพ์ใจ สุขสวัสดิ์",
        "female",
        "1984-04-14 00:00:00",
        "user_004",
        "2026-05-04 14:20:00",
        158.0,
        57.0,
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
    date_of_birth: str
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

    @property
    def age_components(self) -> tuple[int, int, int]:
        return _calculate_age_components(self.date_of_birth, self.updated_at)

    def to_summary(self) -> str:
        age_years, age_months, age_days = self.age_components
        return (
            f"Name: {self.name} | Gender: {self.gender} | "
            f"Age: {age_years}y {age_months}m {age_days}d | "
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


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _calculate_age_components(
    date_of_birth: str, updated_at: str | None = None
) -> tuple[int, int, int]:
    birth_dt = _parse_timestamp(date_of_birth)
    reference_dt = _parse_timestamp(updated_at) if updated_at else datetime.now()

    years = reference_dt.year - birth_dt.year
    months = reference_dt.month - birth_dt.month
    days = reference_dt.day - birth_dt.day

    if reference_dt.time() < birth_dt.time():
        days -= 1

    if days < 0:
        months -= 1
        previous_month = reference_dt.month - 1 or 12
        previous_year = (
            reference_dt.year if reference_dt.month > 1 else reference_dt.year - 1
        )
        days += calendar.monthrange(previous_year, previous_month)[1]

    if months < 0:
        years -= 1
        months += 12

    return years, months, days


def _calculate_age_for_bmr(date_of_birth: str, updated_at: str) -> float:
    age_years, age_months, age_days = _calculate_age_components(
        date_of_birth, updated_at
    )
    return age_years + (age_months / 12) + (age_days / 365)


def _calculate_metrics(
    *,
    gender: str,
    date_of_birth: str,
    updated_at: str,
    height_cm: float,
    weight_kg: float,
) -> tuple[float, float]:
    age = _calculate_age_for_bmr(date_of_birth, updated_at)
    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)

    gender_normalized = gender.strip().lower()
    bmr_offset = 5 if gender_normalized == "male" else -161
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + bmr_offset

    return round(bmi, 1), round(bmr, 1)


def _build_seed_record(seed_record: tuple) -> tuple:
    (
        uuid,
        name,
        gender,
        date_of_birth,
        user_id,
        updated_at,
        height_cm,
        weight_kg,
        hbpm,
        blood_pressure,
        o2,
        body_temp,
    ) = seed_record

    bmi, bmr = _calculate_metrics(
        gender=gender,
        date_of_birth=date_of_birth,
        updated_at=updated_at,
        height_cm=height_cm,
        weight_kg=weight_kg,
    )

    return (
        uuid,
        name,
        gender,
        date_of_birth,
        user_id,
        updated_at,
        height_cm,
        weight_kg,
        bmr,
        bmi,
        hbpm,
        blood_pressure,
        o2,
        body_temp,
    )


def _schema_has_date_of_birth(conn: sqlite3.Connection) -> bool:
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(health)").fetchall()
    }
    if not columns:
        return False
    return "date_of_birth" in columns and "age_years" not in columns


def init_db(db_path: str | Path = HEALTH_DB_PATH) -> None:
    """Create the health table and seed sample data if empty. Safe to call repeatedly."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        table_exists = (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'health'"
            ).fetchone()
            is not None
        )

        if table_exists and not _schema_has_date_of_birth(conn):
            conn.execute("DROP TABLE health")
            conn.execute("DROP INDEX IF EXISTS idx_health_user_updated")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS health (
                uuid           TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                gender         TEXT NOT NULL,
                date_of_birth  TEXT NOT NULL,
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
                    (uuid, name, gender, date_of_birth, user_id, updated_at,
                     height_cm, weight_kg, bmr, bmi, hbpm, blood_pressure, o2, body_temp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [_build_seed_record(record) for record in _SEED_RECORDS],
            )
        conn.commit()


def _row_to_record(row: sqlite3.Row) -> HealthRecord:
    return HealthRecord(
        uuid=row["uuid"],
        name=row["name"],
        gender=row["gender"],
        date_of_birth=row["date_of_birth"],
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


def get_all_health_records(db_path: str | Path = HEALTH_DB_PATH) -> list[HealthRecord]:
    """Return all health records."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM health").fetchall()
    return [_row_to_record(row) for row in rows]


def create_health_record(
    *,
    name: str,
    gender: str,
    date_of_birth: str,
    user_id: str,
    height_cm: float,
    weight_kg: float,
    hbpm: int,
    blood_pressure: str,
    o2: float,
    body_temp: float,
    updated_at: str | None = None,
    db_path: str | Path = HEALTH_DB_PATH,
) -> HealthRecord:
    """Create and persist a health record, returning the inserted row."""

    normalized_updated_at = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized_gender = gender.strip().lower()
    normalized_dob = _parse_timestamp(date_of_birth.strip()).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    bmi, bmr = _calculate_metrics(
        gender=normalized_gender,
        date_of_birth=normalized_dob,
        updated_at=normalized_updated_at,
        height_cm=height_cm,
        weight_kg=weight_kg,
    )

    record = HealthRecord(
        uuid=str(uuid4()),
        name=name.strip(),
        gender=normalized_gender,
        date_of_birth=normalized_dob,
        user_id=user_id.strip(),
        updated_at=normalized_updated_at,
        height_cm=height_cm,
        weight_kg=weight_kg,
        bmr=bmr,
        bmi=bmi,
        hbpm=hbpm,
        blood_pressure=blood_pressure.strip(),
        o2=o2,
        body_temp=body_temp,
    )

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO health
                (uuid, name, gender, date_of_birth, user_id, updated_at,
                 height_cm, weight_kg, bmr, bmi, hbpm, blood_pressure, o2, body_temp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.uuid,
                record.name,
                record.gender,
                record.date_of_birth,
                record.user_id,
                record.updated_at,
                record.height_cm,
                record.weight_kg,
                record.bmr,
                record.bmi,
                record.hbpm,
                record.blood_pressure,
                record.o2,
                record.body_temp,
            ),
        )
        conn.commit()

    return record


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

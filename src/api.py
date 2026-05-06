from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from health_db import (
    HEALTH_DB_PATH,
    create_health_record,
    get_all_health_records,
    get_latest_health,
    init_db,
)


class CreateHealthRecordRequest(BaseModel):
    name: str = Field(min_length=1)
    gender: str = Field(min_length=1)
    age_years: int = Field(ge=0)
    age_months: int = Field(ge=0, le=11)
    age_days: int = Field(ge=0, le=31)
    user_id: str = Field(min_length=1)
    height_cm: float = Field(gt=0)
    weight_kg: float = Field(gt=0)
    hbpm: int = Field(gt=0)
    blood_pressure: str = Field(pattern=r"^\d{2,3}/\d{2,3}$")
    o2: float = Field(gt=0, le=100)
    body_temp: float = Field(gt=0, le=50)


def create_app(*, db_path: str | Path = HEALTH_DB_PATH) -> FastAPI:
    init_db(db_path)

    app = FastAPI(title="Health API")

    @app.get("/health")
    def get_health(
        user_id: Annotated[str, Query(alias="id", min_length=1)],
    ) -> dict:
        record = get_latest_health(user_id, db_path=db_path)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health record not found",
            )
        return asdict(record)

    @app.get("/health/lists")
    def get_all_healths() -> list[dict]:
        return [asdict(record) for record in get_all_health_records(db_path=db_path)]

    @app.post("/health", status_code=status.HTTP_201_CREATED)
    def post_health(payload: CreateHealthRecordRequest) -> dict:
        record = create_health_record(db_path=db_path, **payload.model_dump())
        return asdict(record)

    return app


app = create_app()

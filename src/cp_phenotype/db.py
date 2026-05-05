"""Oracle database connectivity for clinical data schemas.

Handles connection setup via environment variables, TNS configuration,
and basic smoke-testing of the database connection.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import oracledb
import pandas as pd
from dotenv import load_dotenv


@dataclass(frozen=True)
class OracleSettings:
    host: str
    port: str
    service: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        return f"{self.host}:{self.port}/{self.service}"


def load_settings(env_path: str | Path | None = None) -> OracleSettings:
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    required = [
        "ORACLE_HOST",
        "ORACLE_PORT",
        "ORACLE_SERVICE",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing Oracle environment variables: {missing}")

    return OracleSettings(
        host=os.environ["ORACLE_HOST"],
        port=os.environ["ORACLE_PORT"],
        service=os.environ["ORACLE_SERVICE"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
    )


def get_connection(env_path: str | Path | None = None) -> oracledb.Connection:
    settings = load_settings(env_path)
    return oracledb.connect(
        user=settings.user,
        password=settings.password,
        dsn=settings.dsn,
    )


def read_sql(
    sql: str,
    env_path: str | Path | None = None,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    with get_connection(env_path) as conn:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable",
                category=UserWarning,
            )
            return pd.read_sql(sql, conn, params=params)


def smoke_test(env_path: str | Path | None = None) -> dict[str, Any]:
    with get_connection(env_path) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM dual")
            value = cursor.fetchone()[0]
        return {
            "connected": True,
            "server_version": conn.version,
            "smoke_query": int(value),
        }

from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import models  # noqa: F401
from app.database.database import Base, engine


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    schema = f"test_market_{uuid4().hex}"
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    connection = engine.connect()
    try:
        connection.execute(text(f'SET search_path TO "{schema}"'))
        Base.metadata.create_all(bind=connection)
        connection.commit()
        session = Session(bind=connection, expire_on_commit=False)
        try:
            yield session
        finally:
            session.close()
    finally:
        connection.execute(text("SET search_path TO public"))
        connection.commit()
        connection.close()
        with engine.begin() as cleanup_connection:
            cleanup_connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))

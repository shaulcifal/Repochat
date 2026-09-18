"""Engine/session setup. Schema is created via Base.metadata.create_all for
now -- fine while the schema is still moving; swap in Alembic once it settles."""

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from repochat.domain.models import Base

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.environ["DATABASE_URL"]
        _engine = create_engine(url, future=True)
    return _engine


def ensure_schema() -> None:
    Base.metadata.create_all(get_engine())


def get_session() -> Session:
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return factory()

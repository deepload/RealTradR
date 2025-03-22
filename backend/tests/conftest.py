"""
Configuration for pytest

This module contains fixtures and configuration for pytest.
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from unittest.mock import MagicMock, patch

from app.main import app
from app.db.base import Base
from app.api.deps import get_db
from app.core.config import get_settings


# Test database URL
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql://postgres:password@localhost/realtradrtestdb"
)

# Create test engine
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def db_engine():
    """Create a clean database for testing"""
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db(db_engine) -> Generator[Session, None, None]:
    """Get a database session for testing"""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    # Use the session instead of the default one
    app.dependency_overrides[get_db] = lambda: session

    yield session

    # Clean up
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db) -> Generator[TestClient, None, None]:
    """Get a TestClient for testing"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
def mock_current_user() -> Generator[MagicMock, None, None]:
    """Mock the current user for testing authenticated endpoints"""
    with patch("app.api.deps.get_current_user") as mock:
        # Create a mock user
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@example.com"
        mock_user.username = "testuser"
        mock_user.is_active = True
        mock_user.is_superuser = False
        
        # Return the mock user
        mock.return_value = mock_user
        yield mock_user


@pytest.fixture(scope="function")
def mock_ibkr_broker() -> Generator[MagicMock, None, None]:
    """Mock the IBKR broker for testing broker interactions"""
    with patch("app.services.broker_factory.BrokerFactory.create_broker") as mock:
        # Create a mock broker
        mock_broker = MagicMock()
        mock_broker.connect.return_value = True
        mock_broker.disconnect.return_value = None
        mock_broker.is_connected.return_value = True
        
        # Return the mock broker
        mock.return_value = mock_broker
        yield mock_broker

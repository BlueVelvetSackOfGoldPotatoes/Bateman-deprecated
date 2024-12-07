# backend/alembic/env.py

from __future__ import with_statement
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlalchemy import create_engine
from alembic import context

# ---------------------#
# 1. Setup Environment #
# ---------------------#

# Add the backend directory to sys.path to ensure modules can be imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your SQLAlchemy models here
from database import Base  # Import Base from your database.py
import models  # Import all models to ensure Alembic detects them

# ---------------------#
# 2. Configure Alembic #
# ---------------------#

# Retrieve the Alembic Config object, which provides access to the values within the .ini file
config = context.config

# Interpret the config file for Python logging. This sets up loggers defined in the alembic.ini file.
fileConfig(config.config_file_name)

# Set the target metadata for 'autogenerate' support. This is necessary for Alembic to detect model changes.
target_metadata = Base.metadata

# ---------------------#
# 3. Database URL      #
# ---------------------#

def get_database_url():
    """
    Retrieves the DATABASE_URL from environment variables.
    Raises:
        EnvironmentError: If DATABASE_URL is not set.
    Returns:
        str: The database URL.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL environment variable not set.")
    return db_url

# ---------------------#
# 4. Migration Functions #
# ---------------------#

def run_migrations_offline():
    """
    Run migrations in 'offline' mode.
    This configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the given string to the script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Enables type comparison for autogenerate
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """
    Run migrations in 'online' mode.
    In this scenario, an Engine is created and a connection is associated with the context.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_database_url()

    # Create an SQLAlchemy engine using the configuration
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Establish a connection and associate it with the Alembic context
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Enables type comparison for autogenerate
        )

        with context.begin_transaction():
            context.run_migrations()

# ---------------------#
# 5. Determine Mode    #
# ---------------------#

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

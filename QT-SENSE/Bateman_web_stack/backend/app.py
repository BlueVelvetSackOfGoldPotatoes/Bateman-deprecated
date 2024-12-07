# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Import your routers
from routers import lead, personnel

# Import database setup
from database import engine, Base

# Initialize the FastAPI app
app = FastAPI(
    title="BATEMAN API",
    description="API for managing and analyzing leads and personnel.",
    version="1.0.0"
)

# ---------------------#
# 1. CORS Configuration #
# ---------------------#

# List of allowed origins for CORS
# You can add more origins as needed, especially for production
origins = [
    os.getenv("FRONTEND_URL", "http://localhost:3000"),  # Default to React's localhost
    # Add other allowed origins here
    # Example: "https://your-production-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],      # Allows all HTTP methods
    allow_headers=["*"],      # Allows all headers
)

# ---------------------#
# 2. Database Initialization #
# ---------------------#

# Create all tables in the database.
# **Note:** It's recommended to use Alembic for handling migrations in production.
# The following line is suitable for development purposes only.
Base.metadata.create_all(bind=engine)

# ---------------------#
# 3. Include Routers #
# ---------------------#

app.include_router(lead.router)
app.include_router(personnel.router)

# ---------------------#
# 4. Health Check Endpoint #
# ---------------------#

@app.get("/")
def read_root():
    """
    Root endpoint to verify that the API is running.
    """
    return {"message": "Welcome to the BATEMAN API!"}

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify database connectivity and API health.
    """
    try:
        # Simple database connectivity check
        # You can enhance this by performing actual queries if needed
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "details": str(e)}

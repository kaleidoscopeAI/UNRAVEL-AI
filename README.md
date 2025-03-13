markdown
# Unravel AI

Unravel AI (formerly Kaleidoscope AI Pro) is an advanced AI-powered software system generator that creates detailed technical specifications based on user-provided descriptions. It offers multiple subscription tiers, secure authentication, and asynchronous task processing for generating and sanitizing software systems.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Environment Configuration](#environment-configuration)
  - [Database Setup](#database-setup)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Generating a System](#generating-a-system)
  - [Exporting a System](#exporting-a-system)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Features

- **AI-Powered System Generation**: Leverages OpenAI's GPT models to produce detailed technical specifications.
- **Subscription Management**: Offers Basic, Pro, and Enterprise tiers with varying usage limits.
- **Asynchronous Task Processing**: Utilizes Celery for background tasks such as system generation and code sanitization.
- **Secure Authentication**: Implements JWT-based authentication for secure API access.
- **Code Sanitization**: Removes branding and sensitive markers from exported systems (available in Pro and Enterprise tiers).
- **Enterprise Features**: Includes advanced options like SSO and audit logging for Enterprise users.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.10+**
- **PostgreSQL**
- **Redis**
- **Docker** (optional, for containerized deployment)
- **Virtualenv** (recommended for managing dependencies)

## Setup


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash
pip install -r requirements.txt
Set Up Environment Variables:

    Copy the example environment file:
    bash

Database Setup

    Start PostgreSQL and Redis:
        Use Docker Compose to launch the required services:
        bash

    docker-compose up -d db redis

Run Migrations:

    Apply database migrations with Alembic:
    bash

        alembic upgrade head

Running the Application

    Start the Web Server:
    bash

uvicorn app.main:app --host 0.0.0.0 --port 8000
Start the Celery Worker:
bash

    celery -A app.worker.celery worker --loglevel=info
    Access the Application:
        The API will be available at http://localhost:8000.
        Check the health endpoint: http://localhost:8000/health.

Usage
API Endpoints

    Health Check: GET /health
    Generate System: POST /api/v1/systems
    Get System Status: GET /api/v1/systems/{task_id}
    Export System: POST /api/v1/systems/{system_id}/export

Generating a System

To generate a system, send a POST request to /api/v1/systems with a JSON body containing description and complexity.

Example:
json
{
  "description": "A simple todo list application",
  "complexity": 3
}
Exporting a System

To export a sanitized system (Pro and Enterprise tiers only), send a POST request to /api/v1/systems/{system_id}/export.
Testing

Run the test suite to ensure everything is working correctly:
bash
pytest tests/
Deployment

To deploy the application using Docker:

    Build and Start Containers:
    bash

docker-compose up -d --build
Run Migrations:
bash
docker-compose exec web alembic upgrade head
Check Health:
bash

    curl http://localhost:8000/health

For more details, refer to the scripts/deploy.sh script.
Contributing

For more details, see CONTRIBUTING.md.
License

This project is licensed under the MIT License. See LICENSE for details.
Contact



text

### Notes
-

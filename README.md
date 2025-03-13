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
-# Unravel AI End User License Agreement

This End User License Agreement ("Agreement") is a legal agreement between you ("Licensee") and [Your Company Name] ("Licensor") for the use of the Unravel AI software ("Software").

By installing, copying, or otherwise using the Software, you agree to be bound by the terms of this Agreement. If you do not agree to the terms of this Agreement, do not install or use the Software.

## 1. Definitions

- **"Software"** means the Unravel AI software program provided by Licensor.
- **"Licensor"** means [Your Company Name].
- **"Licensee"** means the individual or entity that is using the Software.

## 2. Grant of License

Licensor grants Licensee a non-exclusive, non-transferable license to use the Software for its intended purpose, subject to the terms and conditions of this Agreement.

## 3. Restrictions

Licensee shall not:

a. Redistribute, sell, lease, or otherwise transfer the Software to any third party.  
b. Modify, adapt, or create derivative works of the Software.  
c. Reverse engineer, decompile, or disassemble the Software.  
d. Use the Software for any unlawful purpose.

## 4. Intellectual Property

The Software is protected by copyright and other intellectual property laws. Licensor retains all rights, title, and interest in and to the Software, including all intellectual property rights. This Agreement does not grant Licensee any rights to use Licensor's trademarks or trade names.

## 5. Warranty Disclaimer

THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. LICENSOR DOES NOT WARRANT THAT THE SOFTWARE WILL MEET LICENSEE'S REQUIREMENTS OR THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE.

## 6. Limitation of Liability

IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES, ARISING OUT OF OR IN CONNECTION WITH THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

## 7. Termination

This Agreement is effective until terminated. Licensor may terminate this Agreement immediately if Licensee breaches any term of this Agreement. Upon termination, Licensee must cease all use of the Software and destroy all copies in its possession.

## 8. Governing Law

This Agreement shall be governed by and construed in accordance with the laws of [Your Jurisdiction], without regard to its conflict of law principles.

## 9. Entire Agreement

This Agreement constitutes the entire agreement between the parties and supersedes all prior agreements and understandings, whether written or oral, relating to the subject matter of this Agreement.

---

By using the Software, you acknowledge that you have read this Agreement, understand it, and agree to be bound by its terms.

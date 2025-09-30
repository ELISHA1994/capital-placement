# CV Matching Platform

Intelligent CV/Resume matching system with semantic search capabilities and multi-tenant support.

## Features

- **Semantic Search**: Advanced AI-powered matching beyond keyword matching
- **Multi-tenant Architecture**: Complete data isolation between organizations  
- **Scalable Design**: Handle 2M+ profiles with sub-2 second search response
- **Cloud-Agnostic AI**: Direct OpenAI integration with local development fallbacks
- **Modern Tech Stack**: FastAPI, Python 3.12+, async/await patterns

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy environment file
cp .env.local .env

# Generate secret key and add to .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Set up database (ensure PostgreSQL is running)
# Database URL: postgresql://cv_user:cv_password@localhost:5432/cv-analytic

# Run database migrations
python run_migrations.py

# Start the application
python -m uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for API documentation.

## System Initialization

After running the application for the first time, initialize the system with a super admin:

```bash
# Check system status
curl -X GET "http://localhost:8000/api/v1/setup/status"

# Initialize system with super admin
curl -X POST "http://localhost:8000/api/v1/setup/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@system.local",
    "password": "SuperAdmin123!",
    "full_name": "System Administrator"
  }'
```

**Default Super Admin Credentials:**
- **Email**: `admin@system.local`
- **Password**: `SuperAdmin123!`
- **Role**: `super_admin`
- **Tenant ID**: `00000000-0000-0000-0000-000000000000` (System tenant)

Use these credentials to:
- Create new tenants
- Manage system-wide settings
- Create additional admin users
- Monitor system health

## Documentation

- [Local Setup Guide](docs/LOCAL_SETUP.md) - Detailed setup instructions
- [Architecture Overview](docs/ARCHITECTURE_SOLUTION.md) - System design and patterns
- [Implementation Guide](docs/implementation-guide.md) - Detailed implementation details
- [Deployment Guide](docs/deployment-guide.md) - Cloud deployment instructions

## Development

The system automatically detects your environment and uses appropriate services:
- **Local**: OpenAI API, Memory Cache, Local Files, PostgreSQL
- **Development**: OpenAI API, Redis Cache, Local Files, PostgreSQL
- **Production**: OpenAI API, Redis Cache, Cloud Storage, PostgreSQL

Cloud-agnostic design - same codebase works with any infrastructure!
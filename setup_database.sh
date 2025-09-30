#!/bin/bash

# PostgreSQL Database Setup Script
# Creates cv-analytic database with user and enables pgvector extension

set -e  # Exit on any error

# Configuration
DB_NAME="cv-analytic"
DB_USER="cv_user"
DB_PASSWORD="cv_password"  # Change this to a secure password
POSTGRES_USER="postgres"

echo "🔧 Setting up PostgreSQL database for CV Analytics..."
echo ""
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo ""

# Check if PostgreSQL is in PATH
if ! command -v psql &> /dev/null; then
    echo "⚠️  PostgreSQL commands not in PATH. Adding temporarily..."
    export PATH="/Library/PostgreSQL/16/bin:$PATH"
fi

# Function to run SQL commands
run_sql() {
    local sql="$1"
    PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -c "$sql" 2>&1
}

# Check if PostgreSQL is running
echo "🔍 Checking PostgreSQL connection..."
if ! PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -c '\q' 2>/dev/null; then
    echo "❌ Cannot connect to PostgreSQL."
    echo ""
    echo "Please ensure PostgreSQL is running and you can connect with:"
    echo "  psql -U postgres"
    echo ""
    echo "You may need to enter the postgres user password."
    exit 1
fi

echo "✅ PostgreSQL connection successful"
echo ""

# Check if user already exists
echo "🔍 Checking if user exists..."
USER_EXISTS=$(PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" 2>/dev/null || echo "")

if [ -z "$USER_EXISTS" ]; then
    echo "👤 Creating database user: $DB_USER"
    PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null
    echo "✅ User created successfully"
else
    echo "✅ User already exists"
fi

echo ""

# Check if database already exists
echo "🔍 Checking if database exists..."
DB_EXISTS=$(PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>/dev/null || echo "")

if [ -z "$DB_EXISTS" ]; then
    echo "🗄️  Creating database: $DB_NAME"
    PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -c "CREATE DATABASE \"$DB_NAME\" OWNER $DB_USER;" 2>/dev/null
    echo "✅ Database created successfully"
else
    echo "✅ Database already exists"
    echo "⚠️  Setting owner to $DB_USER"
    PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -c "ALTER DATABASE \"$DB_NAME\" OWNER TO $DB_USER;" 2>/dev/null
fi

echo ""

# Enable pgvector extension
echo "🔌 Enabling pgvector extension..."
PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null
echo "✅ pgvector extension enabled"

echo ""

# Grant privileges
echo "🔐 Setting up permissions..."
PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON DATABASE \"$DB_NAME\" TO $DB_USER;" 2>/dev/null
PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;" 2>/dev/null
echo "✅ Permissions granted"

echo ""

# Verify installation
echo "🔍 Verifying setup..."
EXTENSION_CHECK=$(PGPASSWORD="" psql -U "$POSTGRES_USER" -h localhost -d "$DB_NAME" -tAc "SELECT 1 FROM pg_extension WHERE extname='vector'" 2>/dev/null || echo "")

if [ -z "$EXTENSION_CHECK" ]; then
    echo "❌ pgvector extension not found!"
    exit 1
fi

echo "✅ Verification complete"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Database setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Connection Details:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Database: $DB_NAME"
echo "User:     $DB_USER"
echo "Password: $DB_PASSWORD"
echo "Host:     localhost"
echo "Port:     5432"
echo ""
echo "📝 Connection String:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"
echo ""
echo "🔗 Connect with psql:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "psql -U $DB_USER -d $DB_NAME -h localhost"
echo ""
echo "💡 Test pgvector:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CREATE TABLE test_vectors (id SERIAL PRIMARY KEY, embedding VECTOR(3));"
echo "INSERT INTO test_vectors (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');"
echo "SELECT * FROM test_vectors ORDER BY embedding <-> '[3,1,2]' LIMIT 5;"
echo ""
echo "⚠️  IMPORTANT: Change the password in this script before using in production!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
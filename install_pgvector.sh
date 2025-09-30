#!/bin/bash

# pgvector Installation Script for PostgreSQL 16 on macOS
# This script installs the pgvector extension for your local PostgreSQL installation

set -e  # Exit on any error

echo "🚀 Starting pgvector installation for PostgreSQL 16..."
echo ""

# Set PostgreSQL config path
export PG_CONFIG=/Library/PostgreSQL/16/bin/pg_config

# Verify pg_config is accessible
if [ ! -f "$PG_CONFIG" ]; then
    echo "❌ Error: pg_config not found at $PG_CONFIG"
    exit 1
fi

echo "✅ Found PostgreSQL 16 at: $PG_CONFIG"
echo ""

# Check if Xcode Command Line Tools are installed
if ! xcode-select -p &>/dev/null; then
    echo "📦 Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "⚠️  Please complete the Xcode installation and run this script again."
    exit 0
fi

echo "✅ Xcode Command Line Tools are installed"
echo ""

# Navigate to temp directory
cd /tmp

# Remove existing pgvector directory if it exists
if [ -d "pgvector" ]; then
    echo "🧹 Removing existing pgvector directory..."
    rm -rf pgvector
fi

# Clone pgvector repository
echo "📥 Downloading pgvector v0.8.1..."
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector

echo ""
echo "🔨 Compiling pgvector..."

# Check for SDK issue and create symlink if needed
SDK_PATH="/Library/Developer/CommandLineTools/SDKs"
if [ -d "$SDK_PATH" ]; then
    # Find available SDK
    AVAILABLE_SDK=$(ls -1 "$SDK_PATH" | grep "MacOSX.*\.sdk$" | head -1)

    if [ ! -z "$AVAILABLE_SDK" ]; then
        # Check if the problematic SDK path exists
        if [ ! -e "$SDK_PATH/MacOSX11.3.sdk" ]; then
            echo "⚠️  Creating SDK symlink to fix compilation issue..."
            sudo ln -sf "$SDK_PATH/$AVAILABLE_SDK" "$SDK_PATH/MacOSX11.3.sdk"
            echo "✅ Created symlink: MacOSX11.3.sdk -> $AVAILABLE_SDK"
        fi
    fi
fi

make

echo ""
echo "📦 Installing pgvector (requires sudo)..."
sudo --preserve-env=PG_CONFIG make install

# Clean up the symlink if we created it
if [ -L "$SDK_PATH/MacOSX11.3.sdk" ]; then
    echo "🧹 Cleaning up temporary SDK symlink..."
    sudo rm "$SDK_PATH/MacOSX11.3.sdk"
fi

echo ""
echo "✅ pgvector has been successfully installed!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Add PostgreSQL to your PATH (optional but recommended):"
echo ""
echo "   For zsh (default on newer macOS):"
echo "   echo 'export PATH=\"/Library/PostgreSQL/16/bin:\$PATH\"' >> ~/.zshrc"
echo "   source ~/.zshrc"
echo ""
echo "   For bash:"
echo "   echo 'export PATH=\"/Library/PostgreSQL/16/bin:\$PATH\"' >> ~/.bash_profile"
echo "   source ~/.bash_profile"
echo ""
echo "2. Connect to your database and enable the extension:"
echo ""
echo "   psql -U postgres -d your_database"
echo ""
echo "   Then run:"
echo "   CREATE EXTENSION IF NOT EXISTS vector;"
echo ""
echo "3. Verify installation:"
echo ""
echo "   SELECT * FROM pg_extension WHERE extname = 'vector';"
echo ""
echo "4. Test with a simple example:"
echo ""
echo "   CREATE TABLE items (id SERIAL PRIMARY KEY, embedding VECTOR(3));"
echo "   INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');"
echo "   SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎉 Installation complete! Happy vector searching!"
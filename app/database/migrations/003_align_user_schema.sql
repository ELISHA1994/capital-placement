-- Migration 002: Align user schema with User model
-- Adds missing columns and features to support the User business model

-- Add missing columns to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_superuser BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS roles JSONB DEFAULT '[]';
ALTER TABLE users ADD COLUMN IF NOT EXISTS permissions JSONB DEFAULT '[]';
ALTER TABLE users ADD COLUMN IF NOT EXISTS settings JSONB DEFAULT '{}';

-- Add computed column for full_name (derived from first_name + last_name)
-- This maintains backward compatibility while supporting the User model
ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255);

-- Create function to automatically update full_name when first_name or last_name changes
CREATE OR REPLACE FUNCTION update_user_full_name()
RETURNS TRIGGER AS $$
BEGIN
    -- Construct full_name from first_name and last_name
    NEW.full_name = TRIM(COALESCE(NEW.first_name, '') || ' ' || COALESCE(NEW.last_name, ''));
    
    -- If full_name ends up being just whitespace, set to NULL
    IF LENGTH(TRIM(NEW.full_name)) = 0 THEN
        NEW.full_name = NULL;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically maintain full_name
DROP TRIGGER IF EXISTS update_user_full_name_trigger ON users;
CREATE TRIGGER update_user_full_name_trigger
    BEFORE INSERT OR UPDATE OF first_name, last_name ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_user_full_name();

-- Update existing records to populate full_name
UPDATE users 
SET full_name = TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, ''))
WHERE full_name IS NULL;

-- Update any empty full_name values to NULL
UPDATE users 
SET full_name = NULL 
WHERE LENGTH(TRIM(COALESCE(full_name, ''))) = 0;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_users_is_superuser ON users(is_superuser);
CREATE INDEX IF NOT EXISTS idx_users_is_verified ON users(is_verified);
CREATE INDEX IF NOT EXISTS idx_users_full_name ON users(full_name);

-- Create GIN indexes for JSONB columns for efficient querying
CREATE INDEX IF NOT EXISTS idx_users_roles ON users USING GIN (roles);
CREATE INDEX IF NOT EXISTS idx_users_permissions ON users USING GIN (permissions);
CREATE INDEX IF NOT EXISTS idx_users_settings ON users USING GIN (settings);

-- Add comments for documentation
COMMENT ON COLUMN users.is_superuser IS 'Indicates if user has superuser privileges';
COMMENT ON COLUMN users.is_verified IS 'Indicates if user email has been verified';
COMMENT ON COLUMN users.roles IS 'Array of role names assigned to user (JSONB)';
COMMENT ON COLUMN users.permissions IS 'Array of direct permissions assigned to user (JSONB)';
COMMENT ON COLUMN users.settings IS 'User-specific settings and preferences (JSONB)';
COMMENT ON COLUMN users.full_name IS 'Computed full name derived from first_name + last_name';

-- DOWN migration (for rollback)
-- DOWN

-- Drop the trigger and function
DROP TRIGGER IF EXISTS update_user_full_name_trigger ON users;
DROP FUNCTION IF EXISTS update_user_full_name();

-- Drop indexes
DROP INDEX IF EXISTS idx_users_is_superuser;
DROP INDEX IF EXISTS idx_users_is_verified;
DROP INDEX IF EXISTS idx_users_full_name;
DROP INDEX IF EXISTS idx_users_roles;
DROP INDEX IF EXISTS idx_users_permissions;
DROP INDEX IF EXISTS idx_users_settings;

-- Remove added columns (WARNING: This will lose data!)
-- Commented out to prevent accidental data loss
-- ALTER TABLE users DROP COLUMN IF EXISTS is_superuser;
-- ALTER TABLE users DROP COLUMN IF EXISTS is_verified;
-- ALTER TABLE users DROP COLUMN IF EXISTS roles;
-- ALTER TABLE users DROP COLUMN IF EXISTS permissions;
-- ALTER TABLE users DROP COLUMN IF EXISTS settings;
-- ALTER TABLE users DROP COLUMN IF EXISTS full_name;
-- Migration 007: Add missing is_suspended column
-- The TenantConfiguration model has is_suspended but it was missed in migration 005

-- Add is_suspended column 
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS is_suspended BOOLEAN DEFAULT FALSE;

-- Set NOT NULL constraint after adding with default
ALTER TABLE tenants ALTER COLUMN is_suspended SET NOT NULL;

-- Add index for is_suspended for efficient queries
CREATE INDEX IF NOT EXISTS idx_tenants_is_suspended ON tenants(is_suspended);

-- Add comment for documentation
COMMENT ON COLUMN tenants.is_suspended IS 'Whether tenant is suspended (from TenantConfiguration model)';

-- DOWN migration (for rollback)
-- DOWN

-- Drop index
DROP INDEX IF EXISTS idx_tenants_is_suspended;

-- Remove added column (WARNING: This will lose data!)
-- Commented out to prevent accidental data loss during development
-- ALTER TABLE tenants DROP COLUMN IF EXISTS is_suspended;
-- Migration 006: Add usage_metrics column to match service expectations
-- The service code expects usage_metrics but the model uses current_usage
-- Adding usage_metrics as an alias/duplicate for backward compatibility

-- Add usage_metrics column (JSONB like current_usage)
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS usage_metrics JSONB DEFAULT '{}';

-- Copy data from current_usage to usage_metrics for existing records
UPDATE tenants SET usage_metrics = current_usage WHERE usage_metrics IS NULL OR usage_metrics = '{}';

-- Set NOT NULL constraint after populating data
ALTER TABLE tenants ALTER COLUMN usage_metrics SET NOT NULL;

-- Add GIN index for JSONB column for efficient querying
CREATE INDEX IF NOT EXISTS idx_tenants_usage_metrics ON tenants USING GIN (usage_metrics);

-- Add comment for documentation
COMMENT ON COLUMN tenants.usage_metrics IS 'Current usage metrics and statistics (service compatibility alias)';

-- DOWN migration (for rollback)
-- DOWN

-- Drop index
DROP INDEX IF EXISTS idx_tenants_usage_metrics;

-- Remove added column (WARNING: This will lose data!)
-- Commented out to prevent accidental data loss during development
-- ALTER TABLE tenants DROP COLUMN IF EXISTS usage_metrics;
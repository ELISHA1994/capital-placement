-- Migration 003: Add display_name column to tenants table
-- Adds the missing display_name column that's required by the TenantConfiguration model

-- Add display_name column to tenants table
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS display_name VARCHAR(200);

-- Update existing tenants to have display_name based on name
-- This ensures backward compatibility for existing data
UPDATE tenants 
SET display_name = name 
WHERE display_name IS NULL;

-- For the specific case where name might be a slug/identifier, 
-- let's make display_name more human-readable by capitalizing words
UPDATE tenants 
SET display_name = INITCAP(REPLACE(name, '-', ' '))
WHERE display_name = name AND name ~ '^[a-z0-9_-]+$';

-- Set NOT NULL constraint after populating data
-- This ensures the column matches the model requirement
ALTER TABLE tenants ALTER COLUMN display_name SET NOT NULL;

-- Add index for display_name for efficient queries
CREATE INDEX IF NOT EXISTS idx_tenants_display_name ON tenants(display_name);

-- Add comment for documentation
COMMENT ON COLUMN tenants.display_name IS 'Human-readable display name for tenant shown in UI';

-- Update the primary contact email field to allow longer emails if needed
-- (The model specifies email fields, make sure DB supports them)
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS primary_contact_email VARCHAR(255);

-- Add index for primary_contact_email
CREATE INDEX IF NOT EXISTS idx_tenants_primary_contact_email ON tenants(primary_contact_email);

-- Add comment for primary_contact_email
COMMENT ON COLUMN tenants.primary_contact_email IS 'Primary contact email address for tenant communication';

-- Update subscription-related fields to match the model
-- Add subscription_tier column if it doesn't exist (may already exist as 'tier')
DO $$
BEGIN
    -- Check if subscription_tier column exists, if not rename tier to subscription_tier
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tenants' AND column_name = 'subscription_tier') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tenants' AND column_name = 'tier') THEN
            ALTER TABLE tenants RENAME COLUMN tier TO subscription_tier;
        ELSE
            ALTER TABLE tenants ADD COLUMN subscription_tier VARCHAR(50) DEFAULT 'free';
        END IF;
    END IF;
END $$;

-- Ensure subscription_tier has proper values based on the SubscriptionTier enum
-- Update any invalid values to 'free'
UPDATE tenants 
SET subscription_tier = 'free' 
WHERE subscription_tier NOT IN ('free', 'basic', 'professional', 'enterprise', 'custom');

-- Add check constraint for subscription_tier values
ALTER TABLE tenants DROP CONSTRAINT IF EXISTS chk_subscription_tier;
ALTER TABLE tenants ADD CONSTRAINT chk_subscription_tier 
    CHECK (subscription_tier IN ('free', 'basic', 'professional', 'enterprise', 'custom'));

-- Update index for subscription tier (drop old tier index if it exists)
DROP INDEX IF EXISTS idx_tenants_tier;
CREATE INDEX IF NOT EXISTS idx_tenants_subscription_tier ON tenants(subscription_tier);

-- DOWN migration (for rollback)
-- DOWN

-- Remove constraints and indexes
ALTER TABLE tenants DROP CONSTRAINT IF EXISTS chk_subscription_tier;
DROP INDEX IF EXISTS idx_tenants_display_name;
DROP INDEX IF EXISTS idx_tenants_primary_contact_email;
DROP INDEX IF EXISTS idx_tenants_subscription_tier;

-- Remove added columns (WARNING: This will lose data!)
-- Commented out to prevent accidental data loss during development
-- ALTER TABLE tenants DROP COLUMN IF EXISTS display_name;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS primary_contact_email;

-- Revert subscription_tier back to tier if needed
-- ALTER TABLE tenants RENAME COLUMN subscription_tier TO tier;
-- CREATE INDEX IF NOT EXISTS idx_tenants_tier ON tenants(tier);
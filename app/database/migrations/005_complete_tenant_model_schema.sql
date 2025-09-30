-- Migration 005: Complete tenant model schema alignment  
-- Adds all missing columns from TenantConfiguration model to match database schema exactly

-- Add all missing columns from TenantConfiguration model
-- These are stored as JSONB for complex objects and proper types for simple fields

-- JSONB columns for complex objects
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS quota_limits JSONB DEFAULT '{}';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS search_configuration JSONB DEFAULT '{}';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS processing_configuration JSONB DEFAULT '{}';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS feature_flags JSONB DEFAULT '{}';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS billing_configuration JSONB DEFAULT '{}';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS current_usage JSONB DEFAULT '{}';

-- Simple string/boolean/date columns  
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS index_strategy VARCHAR(50) DEFAULT 'shared';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS dedicated_search_index VARCHAR(255);
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS data_region VARCHAR(100) DEFAULT 'us-central';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS is_system_tenant BOOLEAN DEFAULT FALSE;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS billing_contact_email VARCHAR(255);
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS technical_contact_email VARCHAR(255);
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS timezone VARCHAR(100) DEFAULT 'UTC';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS locale VARCHAR(20) DEFAULT 'en-US';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS date_format VARCHAR(50) DEFAULT 'YYYY-MM-DD';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS suspension_reason TEXT;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS subscription_start_date DATE DEFAULT CURRENT_DATE;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS subscription_end_date DATE;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS description TEXT;

-- Update existing data with default JSONB values for complex fields
UPDATE tenants SET quota_limits = '{}' WHERE quota_limits IS NULL;
UPDATE tenants SET search_configuration = '{}' WHERE search_configuration IS NULL;
UPDATE tenants SET processing_configuration = '{}' WHERE processing_configuration IS NULL;
UPDATE tenants SET feature_flags = '{}' WHERE feature_flags IS NULL;
UPDATE tenants SET billing_configuration = '{}' WHERE billing_configuration IS NULL;
UPDATE tenants SET current_usage = '{}' WHERE current_usage IS NULL;

-- Set NOT NULL constraints for required fields after populating defaults
ALTER TABLE tenants ALTER COLUMN quota_limits SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN search_configuration SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN processing_configuration SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN feature_flags SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN billing_configuration SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN current_usage SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN index_strategy SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN data_region SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN is_system_tenant SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN timezone SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN locale SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN date_format SET NOT NULL;
ALTER TABLE tenants ALTER COLUMN subscription_start_date SET NOT NULL;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_tenants_index_strategy ON tenants(index_strategy);
CREATE INDEX IF NOT EXISTS idx_tenants_data_region ON tenants(data_region);
CREATE INDEX IF NOT EXISTS idx_tenants_is_system_tenant ON tenants(is_system_tenant);
CREATE INDEX IF NOT EXISTS idx_tenants_billing_contact_email ON tenants(billing_contact_email);
CREATE INDEX IF NOT EXISTS idx_tenants_technical_contact_email ON tenants(technical_contact_email);
CREATE INDEX IF NOT EXISTS idx_tenants_timezone ON tenants(timezone);
CREATE INDEX IF NOT EXISTS idx_tenants_subscription_start_date ON tenants(subscription_start_date);
CREATE INDEX IF NOT EXISTS idx_tenants_subscription_end_date ON tenants(subscription_end_date);

-- GIN indexes for JSONB columns for efficient querying
CREATE INDEX IF NOT EXISTS idx_tenants_quota_limits ON tenants USING GIN (quota_limits);
CREATE INDEX IF NOT EXISTS idx_tenants_search_configuration ON tenants USING GIN (search_configuration);
CREATE INDEX IF NOT EXISTS idx_tenants_processing_configuration ON tenants USING GIN (processing_configuration);
CREATE INDEX IF NOT EXISTS idx_tenants_feature_flags ON tenants USING GIN (feature_flags);
CREATE INDEX IF NOT EXISTS idx_tenants_billing_configuration ON tenants USING GIN (billing_configuration);
CREATE INDEX IF NOT EXISTS idx_tenants_current_usage ON tenants USING GIN (current_usage);

-- Add check constraints for valid values
ALTER TABLE tenants DROP CONSTRAINT IF EXISTS chk_index_strategy;
ALTER TABLE tenants ADD CONSTRAINT chk_index_strategy 
    CHECK (index_strategy IN ('shared', 'dedicated', 'hybrid'));

-- Add comments for documentation
COMMENT ON COLUMN tenants.quota_limits IS 'Resource usage quotas and limits (JSONB)';
COMMENT ON COLUMN tenants.search_configuration IS 'Tenant-specific search configuration (JSONB)';
COMMENT ON COLUMN tenants.processing_configuration IS 'Document processing configuration (JSONB)';
COMMENT ON COLUMN tenants.feature_flags IS 'Feature access control flags (JSONB)';
COMMENT ON COLUMN tenants.billing_configuration IS 'Billing and payment configuration (JSONB)';
COMMENT ON COLUMN tenants.current_usage IS 'Current usage metrics and statistics (JSONB)';
COMMENT ON COLUMN tenants.index_strategy IS 'Multi-tenant index isolation strategy';
COMMENT ON COLUMN tenants.dedicated_search_index IS 'Dedicated search index name for this tenant';
COMMENT ON COLUMN tenants.data_region IS 'Data residency region for compliance';
COMMENT ON COLUMN tenants.is_system_tenant IS 'Whether this is the system tenant for super admins';
COMMENT ON COLUMN tenants.billing_contact_email IS 'Billing contact email address';
COMMENT ON COLUMN tenants.technical_contact_email IS 'Technical contact email address';
COMMENT ON COLUMN tenants.timezone IS 'Tenant timezone for date/time display';
COMMENT ON COLUMN tenants.locale IS 'Tenant locale for localization';
COMMENT ON COLUMN tenants.date_format IS 'Preferred date format pattern';
COMMENT ON COLUMN tenants.suspension_reason IS 'Reason for tenant suspension if applicable';
COMMENT ON COLUMN tenants.subscription_start_date IS 'Subscription start date';
COMMENT ON COLUMN tenants.subscription_end_date IS 'Subscription end date (NULL for unlimited)';

-- DOWN migration (for rollback)
-- DOWN

-- Drop indexes
DROP INDEX IF EXISTS idx_tenants_index_strategy;
DROP INDEX IF EXISTS idx_tenants_data_region;
DROP INDEX IF EXISTS idx_tenants_is_system_tenant;
DROP INDEX IF EXISTS idx_tenants_billing_contact_email;
DROP INDEX IF EXISTS idx_tenants_technical_contact_email;
DROP INDEX IF EXISTS idx_tenants_timezone;
DROP INDEX IF EXISTS idx_tenants_subscription_start_date;
DROP INDEX IF EXISTS idx_tenants_subscription_end_date;
DROP INDEX IF EXISTS idx_tenants_quota_limits;
DROP INDEX IF EXISTS idx_tenants_search_configuration;
DROP INDEX IF EXISTS idx_tenants_processing_configuration;
DROP INDEX IF EXISTS idx_tenants_feature_flags;
DROP INDEX IF EXISTS idx_tenants_billing_configuration;
DROP INDEX IF EXISTS idx_tenants_current_usage;

-- Drop constraints
ALTER TABLE tenants DROP CONSTRAINT IF EXISTS chk_index_strategy;

-- Remove added columns (WARNING: This will lose data!)
-- Commented out to prevent accidental data loss during development
-- ALTER TABLE tenants DROP COLUMN IF EXISTS quota_limits;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS search_configuration;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS processing_configuration;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS feature_flags;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS billing_configuration;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS current_usage;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS index_strategy;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS dedicated_search_index;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS data_region;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS is_system_tenant;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS billing_contact_email;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS technical_contact_email;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS timezone;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS locale;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS date_format;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS suspension_reason;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS subscription_start_date;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS subscription_end_date;
-- ALTER TABLE tenants DROP COLUMN IF EXISTS description;
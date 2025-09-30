-- Migration 008: Migrate from single role to roles array
-- Phase 1: Data migration with zero downtime
-- This migration safely moves data from role column to roles column

-- Step 1: Populate roles column with existing role data
-- Only update rows where roles is empty/null and role exists
UPDATE users 
SET roles = JSONB_BUILD_ARRAY(role),
    updated_at = NOW()
WHERE (roles IS NULL OR roles = '[]'::jsonb OR JSONB_ARRAY_LENGTH(roles) = 0)
  AND role IS NOT NULL 
  AND role != '';

-- Step 2: Handle edge cases - users with null/empty role get default 'user' role
UPDATE users 
SET roles = '["user"]'::jsonb,
    updated_at = NOW()
WHERE (roles IS NULL OR roles = '[]'::jsonb OR JSONB_ARRAY_LENGTH(roles) = 0)
  AND (role IS NULL OR role = '');

-- Step 3: Ensure data consistency - verify all users have at least one role
-- This is a safety check to prevent any users without roles
UPDATE users 
SET roles = '["user"]'::jsonb,
    updated_at = NOW()
WHERE roles IS NULL OR roles = '[]'::jsonb OR JSONB_ARRAY_LENGTH(roles) = 0;

-- Step 4: Add validation constraint to ensure roles is never empty
-- This prevents future data integrity issues
ALTER TABLE users ADD CONSTRAINT check_roles_not_empty 
CHECK (roles IS NOT NULL AND JSONB_ARRAY_LENGTH(roles) > 0);

-- Step 5: Create optimized index for roles queries
-- This replaces the need for the old role index
CREATE INDEX IF NOT EXISTS idx_users_roles_gin ON users USING GIN (roles);

-- Step 6: Add comment for documentation
COMMENT ON COLUMN users.roles IS 'User roles array (JSONB) - migrated from single role column';

-- Verification queries (commented out - for manual verification)
-- SELECT role, roles, 
--        CASE 
--          WHEN roles ? role THEN 'CONSISTENT'
--          ELSE 'INCONSISTENT: ' || role || ' not in ' || roles::text
--        END as consistency_check
-- FROM users 
-- WHERE role IS NOT NULL;

-- DOWN migration (rollback strategy)
-- DOWN

-- Step 1: Remove constraint
ALTER TABLE users DROP CONSTRAINT IF EXISTS check_roles_not_empty;

-- Step 2: Populate role column from first role in roles array (if needed for rollback)
-- UPDATE users 
-- SET role = COALESCE(roles->>0, 'user'),
--     updated_at = NOW()
-- WHERE role IS NULL AND JSONB_ARRAY_LENGTH(roles) > 0;

-- Step 3: Drop the new index
DROP INDEX IF EXISTS idx_users_roles_gin;

-- Note: We don't automatically drop the roles column or data
-- This is intentional to prevent data loss during rollback
-- Manual intervention required for complete rollback
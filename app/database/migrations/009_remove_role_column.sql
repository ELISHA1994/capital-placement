-- Migration 009: Remove legacy role column
-- Phase 3: Schema cleanup after successful migration to roles
-- ⚠️ CRITICAL: Only apply after code migration is complete and tested

-- Step 1: Remove the legacy index on role column
DROP INDEX IF EXISTS idx_users_role;

-- Step 2: Add final verification before dropping column
-- Ensure all users have properly populated roles arrays
DO $$
DECLARE
    users_without_roles INTEGER;
    users_with_empty_roles INTEGER;
BEGIN
    -- Check for users without roles
    SELECT COUNT(*) INTO users_without_roles 
    FROM users 
    WHERE roles IS NULL;
    
    -- Check for users with empty roles array
    SELECT COUNT(*) INTO users_with_empty_roles 
    FROM users 
    WHERE roles = '[]'::jsonb OR JSONB_ARRAY_LENGTH(roles) = 0;
    
    -- Fail migration if data integrity issues found
    IF users_without_roles > 0 THEN
        RAISE EXCEPTION 'Migration aborted: % users found with NULL roles', users_without_roles;
    END IF;
    
    IF users_with_empty_roles > 0 THEN
        RAISE EXCEPTION 'Migration aborted: % users found with empty roles array', users_with_empty_roles;
    END IF;
    
    -- Log success
    RAISE NOTICE 'Data integrity verified: All users have valid roles';
END $$;

-- Step 3: Remove the legacy role column
-- ⚠️ POINT OF NO RETURN - Cannot rollback after this step
ALTER TABLE users DROP COLUMN IF EXISTS role;

-- Step 4: Add column comment for documentation
COMMENT ON TABLE users IS 'User authentication table - migrated from single role to multi-role architecture';

-- Step 5: Update system settings to reflect migration completion
INSERT INTO system_settings (key, value, description, category, is_public) 
VALUES (
    'migration.role_to_roles_completed', 
    '"true"', 
    'Indicates successful completion of role to roles migration', 
    'migration', 
    FALSE
) ON CONFLICT (key) DO UPDATE SET 
    value = '"true"',
    updated_at = NOW();

-- Verification query (commented out - for manual verification)
-- SELECT 
--     COUNT(*) as total_users,
--     COUNT(CASE WHEN JSONB_ARRAY_LENGTH(roles) = 1 THEN 1 END) as single_role_users,
--     COUNT(CASE WHEN JSONB_ARRAY_LENGTH(roles) > 1 THEN 1 END) as multi_role_users,
--     AVG(JSONB_ARRAY_LENGTH(roles)) as avg_roles_per_user
-- FROM users;

-- DOWN migration (rollback) - LIMITED CAPABILITY
-- DOWN

-- ⚠️ WARNING: This rollback has LIMITED capability
-- The original role data cannot be fully restored after column drop

-- Step 1: Recreate role column (will be NULL for all users)
-- ALTER TABLE users ADD COLUMN role VARCHAR(50);

-- Step 2: Attempt to populate from first role in roles array
-- UPDATE users 
-- SET role = COALESCE(roles->>0, 'user')
-- WHERE role IS NULL;

-- Step 3: Recreate the index
-- CREATE INDEX idx_users_role ON users(role);

-- Step 4: Remove migration completion flag
-- DELETE FROM system_settings WHERE key = 'migration.role_to_roles_completed';

-- Note: This rollback is destructive and should only be used in emergency situations
-- Multi-role assignments will be lost and reduced to single role
-- Consider data export/import strategy for full rollback capability
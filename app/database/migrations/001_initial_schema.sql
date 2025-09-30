-- Initial schema for CV Matching Platform
-- Includes pgvector extension and core tables

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =======================
-- TENANT MANAGEMENT
-- =======================

-- Tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    tier VARCHAR(50) DEFAULT 'basic',
    max_users INTEGER DEFAULT 10,
    max_jobs INTEGER DEFAULT 100,
    max_candidates INTEGER DEFAULT 1000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_slug CHECK (slug ~ '^[a-z0-9_-]+$')
);

-- Create indexes for tenants
CREATE INDEX idx_tenants_slug ON tenants(slug);
CREATE INDEX idx_tenants_is_active ON tenants(is_active);
CREATE INDEX idx_tenants_tier ON tenants(tier);

-- =======================
-- USER MANAGEMENT
-- =======================

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    role VARCHAR(50) DEFAULT 'user',
    last_login_at TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, email)
);

-- Create indexes for users
CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_active ON users(is_active);
CREATE INDEX idx_users_role ON users(role);

-- User sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(255),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    refresh_expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for user sessions
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token_hash ON user_sessions(token_hash);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_user_sessions_is_active ON user_sessions(is_active);

-- =======================
-- JOB MANAGEMENT
-- =======================

-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    created_by UUID NOT NULL REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    requirements TEXT,
    location VARCHAR(255),
    job_type VARCHAR(50) DEFAULT 'full_time',
    salary_min INTEGER,
    salary_max INTEGER,
    currency VARCHAR(3) DEFAULT 'USD',
    remote_allowed BOOLEAN DEFAULT FALSE,
    status VARCHAR(50) DEFAULT 'active',
    priority INTEGER DEFAULT 0,
    department VARCHAR(100),
    experience_level VARCHAR(50),
    skills_required TEXT[],
    deadline TIMESTAMP WITH TIME ZONE,
    
    -- Vector embeddings for similarity search
    title_embedding VECTOR(1536),
    description_embedding VECTOR(1536),
    requirements_embedding VECTOR(1536),
    combined_embedding VECTOR(1536),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for jobs
CREATE INDEX idx_jobs_tenant_id ON jobs(tenant_id);
CREATE INDEX idx_jobs_created_by ON jobs(created_by);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_job_type ON jobs(job_type);
CREATE INDEX idx_jobs_experience_level ON jobs(experience_level);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);

-- Vector similarity indexes for jobs
CREATE INDEX idx_jobs_title_embedding ON jobs USING ivfflat (title_embedding vector_cosine_ops);
CREATE INDEX idx_jobs_description_embedding ON jobs USING ivfflat (description_embedding vector_cosine_ops);
CREATE INDEX idx_jobs_requirements_embedding ON jobs USING ivfflat (requirements_embedding vector_cosine_ops);
CREATE INDEX idx_jobs_combined_embedding ON jobs USING ivfflat (combined_embedding vector_cosine_ops);

-- =======================
-- CANDIDATE MANAGEMENT
-- =======================

-- Candidates table
CREATE TABLE candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    created_by UUID NOT NULL REFERENCES users(id),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(50),
    location VARCHAR(255),
    current_job_title VARCHAR(255),
    current_company VARCHAR(255),
    experience_years INTEGER,
    expected_salary INTEGER,
    salary_currency VARCHAR(3) DEFAULT 'USD',
    availability VARCHAR(50) DEFAULT 'available',
    notice_period VARCHAR(50),
    skills TEXT[],
    education_level VARCHAR(50),
    languages TEXT[],
    remote_preference BOOLEAN DEFAULT FALSE,
    
    -- CV content and embeddings
    cv_text TEXT,
    cv_summary TEXT,
    cv_embedding VECTOR(1536),
    skills_embedding VECTOR(1536),
    experience_embedding VECTOR(1536),
    combined_embedding VECTOR(1536),
    
    -- File information
    cv_file_name VARCHAR(255),
    cv_file_size INTEGER,
    cv_file_type VARCHAR(50),
    cv_processed_at TIMESTAMP WITH TIME ZONE,
    
    status VARCHAR(50) DEFAULT 'active',
    source VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, email)
);

-- Create indexes for candidates
CREATE INDEX idx_candidates_tenant_id ON candidates(tenant_id);
CREATE INDEX idx_candidates_created_by ON candidates(created_by);
CREATE INDEX idx_candidates_email ON candidates(email);
CREATE INDEX idx_candidates_status ON candidates(status);
CREATE INDEX idx_candidates_availability ON candidates(availability);
CREATE INDEX idx_candidates_experience_years ON candidates(experience_years);
CREATE INDEX idx_candidates_created_at ON candidates(created_at);

-- Vector similarity indexes for candidates
CREATE INDEX idx_candidates_cv_embedding ON candidates USING ivfflat (cv_embedding vector_cosine_ops);
CREATE INDEX idx_candidates_skills_embedding ON candidates USING ivfflat (skills_embedding vector_cosine_ops);
CREATE INDEX idx_candidates_experience_embedding ON candidates USING ivfflat (experience_embedding vector_cosine_ops);
CREATE INDEX idx_candidates_combined_embedding ON candidates USING ivfflat (combined_embedding vector_cosine_ops);

-- =======================
-- CV MATCHING
-- =======================

-- Job-Candidate matches table
CREATE TABLE job_candidate_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_id UUID NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    
    -- Similarity scores
    overall_score DECIMAL(5,4) NOT NULL,
    title_similarity DECIMAL(5,4),
    description_similarity DECIMAL(5,4),
    requirements_similarity DECIMAL(5,4),
    skills_match_score DECIMAL(5,4),
    experience_score DECIMAL(5,4),
    
    -- Match details
    matched_skills TEXT[],
    missing_skills TEXT[],
    experience_gap INTEGER, -- Years difference
    salary_compatibility BOOLEAN,
    location_match BOOLEAN,
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending',
    reviewer_id UUID REFERENCES users(id),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(job_id, candidate_id)
);

-- Create indexes for matches
CREATE INDEX idx_matches_job_id ON job_candidate_matches(job_id);
CREATE INDEX idx_matches_candidate_id ON job_candidate_matches(candidate_id);
CREATE INDEX idx_matches_overall_score ON job_candidate_matches(overall_score DESC);
CREATE INDEX idx_matches_status ON job_candidate_matches(status);
CREATE INDEX idx_matches_created_at ON job_candidate_matches(created_at);

-- =======================
-- MATCHING ANALYTICS
-- =======================

-- Match analytics table
CREATE TABLE match_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    
    -- Analytics data
    match_type VARCHAR(50) NOT NULL, -- 'job_search', 'candidate_search', 'bulk_match'
    search_query TEXT,
    filters_applied JSONB,
    results_count INTEGER,
    top_score DECIMAL(5,4),
    avg_score DECIMAL(5,4),
    processing_time_ms INTEGER,
    
    -- Model information
    embedding_model VARCHAR(100),
    similarity_threshold DECIMAL(5,4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for analytics
CREATE INDEX idx_match_analytics_tenant_id ON match_analytics(tenant_id);
CREATE INDEX idx_match_analytics_job_id ON match_analytics(job_id);
CREATE INDEX idx_match_analytics_candidate_id ON match_analytics(candidate_id);
CREATE INDEX idx_match_analytics_match_type ON match_analytics(match_type);
CREATE INDEX idx_match_analytics_created_at ON match_analytics(created_at);

-- =======================
-- FILE MANAGEMENT
-- =======================

-- File uploads table
CREATE TABLE file_uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    uploaded_by UUID NOT NULL REFERENCES users(id),
    
    -- File information
    file_name VARCHAR(255) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    
    -- Storage information
    storage_provider VARCHAR(50) DEFAULT 'local',
    storage_path TEXT NOT NULL,
    storage_metadata JSONB,
    
    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'uploaded',
    processing_error TEXT,
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Related entities
    entity_type VARCHAR(50), -- 'candidate', 'job', etc.
    entity_id UUID,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for file uploads
CREATE INDEX idx_file_uploads_tenant_id ON file_uploads(tenant_id);
CREATE INDEX idx_file_uploads_uploaded_by ON file_uploads(uploaded_by);
CREATE INDEX idx_file_uploads_file_hash ON file_uploads(file_hash);
CREATE INDEX idx_file_uploads_entity_type_id ON file_uploads(entity_type, entity_id);
CREATE INDEX idx_file_uploads_processing_status ON file_uploads(processing_status);

-- =======================
-- AUDIT AND LOGGING
-- =======================

-- Audit log table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Action details
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    action VARCHAR(50) NOT NULL, -- 'create', 'update', 'delete', 'view'
    old_values JSONB,
    new_values JSONB,
    
    -- Request context
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(255),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit logs
CREATE INDEX idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_entity_type_id ON audit_logs(entity_type, entity_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

-- =======================
-- SYSTEM CONFIGURATION
-- =======================

-- System settings table
CREATE TABLE system_settings (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    category VARCHAR(100),
    is_public BOOLEAN DEFAULT FALSE,
    updated_by UUID REFERENCES users(id),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default system settings
INSERT INTO system_settings (key, value, description, category, is_public) VALUES
('matching.similarity_threshold', '0.7', 'Default similarity threshold for CV matching', 'matching', FALSE),
('matching.max_results', '50', 'Maximum number of results to return per search', 'matching', FALSE),
('matching.embedding_model', '"text-embedding-ada-002"', 'Default embedding model for text analysis', 'matching', FALSE),
('files.max_upload_size', '10485760', 'Maximum file upload size in bytes (10MB)', 'files', FALSE),
('files.allowed_types', '["pdf", "doc", "docx", "txt"]', 'Allowed file types for CV uploads', 'files', FALSE);

-- =======================
-- FUNCTIONS AND TRIGGERS
-- =======================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_candidates_updated_at BEFORE UPDATE ON candidates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_matches_updated_at BEFORE UPDATE ON job_candidate_matches FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_file_uploads_updated_at BEFORE UPDATE ON file_uploads FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate vector similarity
CREATE OR REPLACE FUNCTION calculate_cosine_similarity(vec1 VECTOR, vec2 VECTOR)
RETURNS DECIMAL(5,4) AS $$
BEGIN
    IF vec1 IS NULL OR vec2 IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN 1 - (vec1 <=> vec2);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- =======================
-- INITIAL DATA
-- =======================

-- Create default tenant for single-tenant mode
INSERT INTO tenants (id, name, slug, description, tier, is_active) 
VALUES (
    uuid_generate_v4(),
    'Default Organization',
    'default',
    'Default tenant for single-tenant deployments',
    'enterprise',
    TRUE
) ON CONFLICT (slug) DO NOTHING;

-- DOWN migration (for rollback)
-- DOWN

-- Drop triggers
DROP TRIGGER IF EXISTS update_tenants_updated_at ON tenants;
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_jobs_updated_at ON jobs;
DROP TRIGGER IF EXISTS update_candidates_updated_at ON candidates;
DROP TRIGGER IF EXISTS update_matches_updated_at ON job_candidate_matches;
DROP TRIGGER IF EXISTS update_file_uploads_updated_at ON file_uploads;

-- Drop functions
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP FUNCTION IF EXISTS calculate_cosine_similarity(VECTOR, VECTOR);

-- Drop tables in reverse order of creation
DROP TABLE IF EXISTS system_settings;
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS file_uploads;
DROP TABLE IF EXISTS match_analytics;
DROP TABLE IF EXISTS job_candidate_matches;
DROP TABLE IF EXISTS candidates;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS user_sessions;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS tenants;
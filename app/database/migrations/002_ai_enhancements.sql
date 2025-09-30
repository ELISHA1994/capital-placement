-- AI Enhancements Migration
-- Adds AI-powered features for semantic search, caching, and document processing

-- =======================
-- EMBEDDINGS STORAGE
-- =======================

-- Embeddings table for storing vector representations
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL,
    embedding_vector vector(1536), -- Default for text-embedding-3-small or text-embedding-ada-002
    content_hash VARCHAR(64), -- SHA-256 hash for deduplication
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique embedding per entity per tenant
    CONSTRAINT unique_entity_embedding UNIQUE (entity_id, entity_type, tenant_id)
);

-- Create indexes for efficient vector search
CREATE INDEX idx_embeddings_entity ON embeddings(entity_id, entity_type);
CREATE INDEX idx_embeddings_tenant ON embeddings(tenant_id);
CREATE INDEX idx_embeddings_model ON embeddings(embedding_model);
CREATE INDEX idx_embeddings_hash ON embeddings(content_hash);
CREATE INDEX idx_embeddings_created_at ON embeddings(created_at);

-- Vector similarity search index using pgvector with HNSW (supports higher dimensions)
-- Note: HNSW is more suitable for high-dimensional vectors like text-embedding-3-large (3072 dimensions)
CREATE INDEX idx_embeddings_vector_cosine ON embeddings USING hnsw (embedding_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- L2 distance index for alternative similarity metrics using HNSW
CREATE INDEX idx_embeddings_vector_l2 ON embeddings USING hnsw (embedding_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

-- =======================
-- SEMANTIC SEARCH CACHE
-- =======================

-- Search cache table for storing query results with semantic similarity
CREATE TABLE search_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL, -- SHA-256 hash of normalized query
    query_text TEXT NOT NULL,
    query_embedding vector(1536),
    results JSONB NOT NULL, -- Cached search results
    result_count INTEGER DEFAULT 0,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    search_type VARCHAR(50) DEFAULT 'semantic', -- semantic, hybrid, text
    filters JSONB DEFAULT '{}', -- Applied search filters
    hit_count INTEGER DEFAULT 0, -- Cache usage tracking
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient cache lookup
CREATE INDEX idx_search_cache_query_hash ON search_cache(query_hash);
CREATE INDEX idx_search_cache_tenant ON search_cache(tenant_id);
CREATE INDEX idx_search_cache_expires ON search_cache(expires_at);
CREATE INDEX idx_search_cache_type ON search_cache(search_type);
CREATE INDEX idx_search_cache_hit_count ON search_cache(hit_count DESC);

-- Vector similarity index for semantic cache matching using HNSW
CREATE INDEX idx_search_cache_vector_cosine ON search_cache USING hnsw (query_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =======================
-- DOCUMENT PROCESSING
-- =======================

-- Document processing status and metadata
CREATE TABLE document_processing (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL, -- References documents.id or similar
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    processing_type VARCHAR(50) NOT NULL, -- pdf_extraction, ai_analysis, embedding_generation
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    input_metadata JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    error_details JSONB DEFAULT '{}',
    processing_duration_ms INTEGER,
    ai_model_used VARCHAR(100),
    token_usage JSONB DEFAULT '{}', -- Track AI token consumption
    quality_score DECIMAL(5,2), -- 0-100 quality assessment
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for document processing tracking
CREATE INDEX idx_document_processing_document ON document_processing(document_id);
CREATE INDEX idx_document_processing_tenant ON document_processing(tenant_id);
CREATE INDEX idx_document_processing_status ON document_processing(status);
CREATE INDEX idx_document_processing_type ON document_processing(processing_type);
CREATE INDEX idx_document_processing_created_at ON document_processing(created_at);
CREATE INDEX idx_document_processing_quality ON document_processing(quality_score DESC);

-- =======================
-- QUERY EXPANSIONS
-- =======================

-- Cache for AI-generated query expansions
CREATE TABLE query_expansions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_query TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL, -- SHA-256 hash for deduplication
    expanded_terms JSONB NOT NULL, -- Array of expanded search terms
    primary_skills JSONB DEFAULT '[]', -- Identified skills
    job_roles JSONB DEFAULT '[]', -- Relevant job titles
    experience_level VARCHAR(50), -- junior, mid, senior, lead
    industry VARCHAR(100),
    confidence_score DECIMAL(5,2), -- AI confidence in expansion
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    ai_model_used VARCHAR(100),
    token_usage JSONB DEFAULT '{}',
    usage_count INTEGER DEFAULT 0, -- Track popularity
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for query expansion lookup
CREATE INDEX idx_query_expansions_hash ON query_expansions(query_hash);
CREATE INDEX idx_query_expansions_tenant ON query_expansions(tenant_id);
CREATE INDEX idx_query_expansions_expires ON query_expansions(expires_at);
CREATE INDEX idx_query_expansions_usage ON query_expansions(usage_count DESC);
CREATE INDEX idx_query_expansions_confidence ON query_expansions(confidence_score DESC);

-- =======================
-- AI ANALYTICS
-- =======================

-- AI operation analytics and monitoring
CREATE TABLE ai_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    operation_type VARCHAR(50) NOT NULL, -- embedding, chat, query_expansion, etc.
    ai_model VARCHAR(100) NOT NULL,
    token_usage JSONB DEFAULT '{}', -- prompt_tokens, completion_tokens, total_tokens
    processing_time_ms INTEGER,
    input_size INTEGER, -- Characters or tokens
    output_size INTEGER, -- Characters or tokens  
    success BOOLEAN DEFAULT TRUE,
    error_type VARCHAR(100), -- If success = FALSE
    cost_estimate DECIMAL(10,6), -- Estimated API cost
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(100), -- Group related operations
    request_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for analytics queries
CREATE INDEX idx_ai_analytics_tenant ON ai_analytics(tenant_id);
CREATE INDEX idx_ai_analytics_operation ON ai_analytics(operation_type);
CREATE INDEX idx_ai_analytics_model ON ai_analytics(ai_model);
CREATE INDEX idx_ai_analytics_success ON ai_analytics(success);
CREATE INDEX idx_ai_analytics_created_at ON ai_analytics(created_at);
CREATE INDEX idx_ai_analytics_session ON ai_analytics(session_id);
CREATE INDEX idx_ai_analytics_user ON ai_analytics(user_id);

-- Composite index for common analytics queries
CREATE INDEX idx_ai_analytics_tenant_date ON ai_analytics(tenant_id, created_at);
CREATE INDEX idx_ai_analytics_operation_date ON ai_analytics(operation_type, created_at);

-- =======================
-- SEMANTIC RELATIONSHIPS
-- =======================

-- Track semantic relationships between entities
CREATE TABLE semantic_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id VARCHAR(255) NOT NULL,
    source_entity_type VARCHAR(50) NOT NULL,
    target_entity_id VARCHAR(255) NOT NULL,
    target_entity_type VARCHAR(50) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL, -- similarity, related, derived, etc.
    similarity_score DECIMAL(8,6) NOT NULL, -- 0.0 to 1.0
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}',
    discovered_by VARCHAR(50), -- ai_analysis, user_action, system_inference
    confidence DECIMAL(5,2), -- 0-100
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Prevent duplicate relationships
    CONSTRAINT unique_relationship UNIQUE (source_entity_id, source_entity_type, target_entity_id, target_entity_type, relationship_type, tenant_id)
);

-- Create indexes for relationship queries
CREATE INDEX idx_relationships_source ON semantic_relationships(source_entity_id, source_entity_type);
CREATE INDEX idx_relationships_target ON semantic_relationships(target_entity_id, target_entity_type);
CREATE INDEX idx_relationships_type ON semantic_relationships(relationship_type);
CREATE INDEX idx_relationships_similarity ON semantic_relationships(similarity_score DESC);
CREATE INDEX idx_relationships_tenant ON semantic_relationships(tenant_id);

-- =======================
-- CONFIGURATION UPDATES
-- =======================

-- Add AI configuration to tenants table
ALTER TABLE tenants ADD COLUMN ai_settings JSONB DEFAULT '{}';

-- Add AI-related user preferences
ALTER TABLE users ADD COLUMN ai_preferences JSONB DEFAULT '{}';

-- =======================
-- ENHANCED PROFILES
-- =======================

-- Add AI-enhanced fields to profiles (if profiles table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'profiles') THEN
        -- Add AI-generated summary
        ALTER TABLE profiles ADD COLUMN ai_summary TEXT;
        
        -- Add AI-extracted skills with confidence scores
        ALTER TABLE profiles ADD COLUMN ai_skills JSONB DEFAULT '{}';
        
        -- Add AI quality assessment
        ALTER TABLE profiles ADD COLUMN quality_assessment JSONB DEFAULT '{}';
        
        -- Add semantic tags
        ALTER TABLE profiles ADD COLUMN semantic_tags JSONB DEFAULT '[]';
        
        -- Add AI processing status
        ALTER TABLE profiles ADD COLUMN ai_processed_at TIMESTAMP WITH TIME ZONE;
        ALTER TABLE profiles ADD COLUMN ai_processing_version VARCHAR(50);
        
        -- Create indexes for AI fields
        CREATE INDEX idx_profiles_ai_processed ON profiles(ai_processed_at);
        CREATE INDEX idx_profiles_ai_version ON profiles(ai_processing_version);
    END IF;
END $$;

-- =======================
-- ENHANCED JOBS
-- =======================

-- Add AI-enhanced fields to jobs table (if it exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'jobs') THEN
        -- Add AI-extracted requirements
        ALTER TABLE jobs ADD COLUMN ai_requirements JSONB DEFAULT '{}';
        
        -- Add AI-generated matching criteria
        ALTER TABLE jobs ADD COLUMN matching_criteria JSONB DEFAULT '{}';
        
        -- Add semantic job tags
        ALTER TABLE jobs ADD COLUMN semantic_tags JSONB DEFAULT '[]';
        
        -- Add AI processing status
        ALTER TABLE jobs ADD COLUMN ai_processed_at TIMESTAMP WITH TIME ZONE;
        ALTER TABLE jobs ADD COLUMN ai_processing_version VARCHAR(50);
        
        -- Create indexes
        CREATE INDEX idx_jobs_ai_processed ON jobs(ai_processed_at);
        CREATE INDEX idx_jobs_ai_version ON jobs(ai_processing_version);
    END IF;
END $$;

-- =======================
-- PERFORMANCE VIEWS
-- =======================

-- View for embedding statistics
CREATE VIEW embedding_stats AS
SELECT 
    tenant_id,
    entity_type,
    embedding_model,
    COUNT(*) as total_embeddings,
    AVG(CASE WHEN content_hash IS NOT NULL THEN 1 ELSE 0 END) as deduplication_rate,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created
FROM embeddings
GROUP BY tenant_id, entity_type, embedding_model;

-- View for search cache performance
CREATE VIEW search_cache_stats AS
SELECT 
    tenant_id,
    search_type,
    COUNT(*) as total_cached_queries,
    AVG(hit_count) as avg_hit_count,
    SUM(hit_count) as total_hits,
    AVG(result_count) as avg_results_per_query,
    COUNT(CASE WHEN expires_at > NOW() THEN 1 END) as active_cache_entries
FROM search_cache
GROUP BY tenant_id, search_type;

-- View for AI operation costs and usage
CREATE VIEW ai_usage_summary AS
SELECT 
    tenant_id,
    operation_type,
    ai_model,
    DATE(created_at) as usage_date,
    COUNT(*) as operation_count,
    SUM(COALESCE((token_usage->>'total_tokens')::INTEGER, 0)) as total_tokens,
    SUM(COALESCE(cost_estimate, 0)) as estimated_cost,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN success THEN 1 END)::DECIMAL / COUNT(*) as success_rate
FROM ai_analytics
GROUP BY tenant_id, operation_type, ai_model, DATE(created_at);

-- =======================
-- CLEANUP FUNCTIONS
-- =======================

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Clean up expired search cache
    DELETE FROM search_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up expired query expansions
    DELETE FROM query_expansions WHERE expires_at < NOW();
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update cache hit counts
CREATE OR REPLACE FUNCTION increment_cache_hit(cache_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE search_cache 
    SET hit_count = hit_count + 1,
        last_accessed = NOW()
    WHERE id = cache_id;
END;
$$ LANGUAGE plpgsql;

-- =======================
-- MIGRATION METADATA
-- =======================

-- Migration tracking is handled automatically by the MigrationManager
-- This migration will be recorded in schema_migrations table upon successful completion

-- =======================
-- COMMENTS FOR DOCUMENTATION
-- =======================

COMMENT ON TABLE embeddings IS 'Stores vector embeddings for entities with pgvector support';
COMMENT ON TABLE search_cache IS 'Caches search results with semantic similarity matching';
COMMENT ON TABLE document_processing IS 'Tracks AI document processing operations and status';
COMMENT ON TABLE query_expansions IS 'Caches AI-generated query expansions for improved search';
COMMENT ON TABLE ai_analytics IS 'Analytics and monitoring for AI operations';
COMMENT ON TABLE semantic_relationships IS 'Tracks semantic relationships between entities';

COMMENT ON INDEX idx_embeddings_vector_cosine IS 'HNSW index for cosine similarity vector search (high-dimensional vectors)';
COMMENT ON INDEX idx_embeddings_vector_l2 IS 'HNSW index for L2 distance vector search (high-dimensional vectors)';

-- Grant appropriate permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cv_app_user;
-- GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO cv_app_user;
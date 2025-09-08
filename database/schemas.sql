-- Skin Allergy Prediction Database Schema
-- Created for user management and search history

-- Users table for authentication and profile management
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- Hashed password for security
    full_name VARCHAR(100),
    date_of_birth DATE,
    gender VARCHAR(10),
    phone_number VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE
);

-- User profiles for extended information
CREATE TABLE IF NOT EXISTS user_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    skin_type VARCHAR(20),
    occupation VARCHAR(50),
    medical_history TEXT,  -- JSON string for medical history
    known_allergies TEXT,  -- JSON string for known allergies
    emergency_contact_name VARCHAR(100),
    emergency_contact_phone VARCHAR(20),
    preferred_language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    profile_picture_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Prediction history - stores all user predictions
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id VARCHAR(50),  -- For anonymous users or session tracking
    
    -- Input data (all the form inputs)
    age INTEGER,
    gender VARCHAR(20),
    skin_type VARCHAR(50),
    occupation VARCHAR(100),
    
    -- Medical history (binary flags)
    family_history_allergies BOOLEAN,
    previous_allergic_reactions BOOLEAN,
    asthma BOOLEAN,
    eczema BOOLEAN,
    autoimmune_conditions BOOLEAN,
    
    -- Environmental factors
    season VARCHAR(20),
    known_primary_allergen VARCHAR(50),
    pollen_count REAL,
    humidity_percent REAL,
    temperature_celsius REAL,
    air_quality_index REAL,
    uv_index REAL,
    
    -- Lifestyle factors
    stress_level INTEGER,
    sleep_quality_score INTEGER,
    exercise_frequency_per_week INTEGER,
    diet_type VARCHAR(30),
    smoking_status VARCHAR(20),
    alcohol_consumption_per_week INTEGER,
    
    -- Product usage
    cosmetic_usage_frequency INTEGER,
    skincare_routine_frequency INTEGER,
    hair_product_usage INTEGER,
    fragrance_usage_frequency INTEGER,
    household_chemical_exposure INTEGER,
    occupational_chemical_exposure INTEGER,
    new_products_tried_recently BOOLEAN,
    
    -- Prediction results
    predicted_risk_level VARCHAR(20),  -- High, Medium, Low
    prediction_confidence REAL,       -- 0.0 to 1.0
    low_risk_probability REAL,
    medium_risk_probability REAL,
    high_risk_probability REAL,
    
    -- Metadata
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20),
    ip_address VARCHAR(45),
    user_agent TEXT,
    
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- User sessions for tracking active sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Favorites/Bookmarks for users to save specific predictions
CREATE TABLE IF NOT EXISTS prediction_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    prediction_id INTEGER NOT NULL,
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES prediction_history (id) ON DELETE CASCADE,
    UNIQUE(user_id, prediction_id)
);

-- Search history for tracking user search patterns
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    session_id VARCHAR(50),
    search_query TEXT,
    search_type VARCHAR(50),  -- 'prediction', 'history', 'filter', etc.
    search_filters TEXT,      -- JSON string for search filters
    results_count INTEGER,
    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Feedback/ratings for predictions
CREATE TABLE IF NOT EXISTS prediction_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    prediction_id INTEGER NOT NULL,
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    is_helpful BOOLEAN,
    actual_outcome VARCHAR(20),  -- If user reports actual allergy outcome
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES prediction_history (id) ON DELETE CASCADE
);

-- System settings and configurations
CREATE TABLE IF NOT EXISTS app_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT,
    setting_type VARCHAR(20) DEFAULT 'string',  -- string, integer, boolean, json
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for tracking important system events
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(50),
    record_id INTEGER,
    old_values TEXT,  -- JSON string
    new_values TEXT,  -- JSON string
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX IF NOT EXISTS idx_prediction_history_user_id ON prediction_history(user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_history_timestamp ON prediction_history(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_history_risk_level ON prediction_history(predicted_risk_level);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history(user_id);
CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history(search_timestamp);

CREATE INDEX IF NOT EXISTS idx_prediction_feedback_prediction_id ON prediction_feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_prediction_feedback_rating ON prediction_feedback(rating);

-- Insert default app settings
INSERT OR IGNORE INTO app_settings (setting_key, setting_value, setting_type, description, is_public) VALUES
('app_name', 'Skin Allergy Risk Predictor', 'string', 'Application name', TRUE),
('app_version', '1.0.0', 'string', 'Current application version', TRUE),
('max_predictions_per_user', '100', 'integer', 'Maximum predictions per user per day', FALSE),
('enable_user_registration', 'true', 'boolean', 'Allow new user registrations', TRUE),
('require_email_verification', 'false', 'boolean', 'Require email verification for new accounts', FALSE),
('session_timeout_hours', '24', 'integer', 'Session timeout in hours', FALSE),
('default_timezone', 'UTC', 'string', 'Default timezone for the application', TRUE);

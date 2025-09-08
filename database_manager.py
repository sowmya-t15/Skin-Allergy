"""
Database Manager for Skin Allergy Risk Prediction System
Handles all database operations including user management and prediction history
"""

import sqlite3
import hashlib
import secrets
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from contextlib import contextmanager
import pandas as pd

class DatabaseManager:
    """Manages all database operations for the Skin Allergy Risk Prediction System"""
    
    def __init__(self, db_path: str = "database/skin_allergy_app.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """Initialize database with schema"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Read and execute schema file
                schema_path = "database/schemas.sql"
                try:
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                    cursor.executescript(schema_sql)
                    self.logger.info("Database schema initialized successfully")
                except FileNotFoundError:
                    self.logger.error(f"Schema file not found: {schema_path}")
                    self.create_tables_programmatically(cursor)
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def create_tables_programmatically(self, cursor):
        """Fallback method to create tables if schema file is missing"""
        tables = {
            'users': '''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''',
            'prediction_history': '''
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    input_data TEXT NOT NULL,
                    predicted_risk_level VARCHAR(20),
                    prediction_confidence REAL,
                    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            '''
        }
        
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            self.logger.info(f"Created table: {table_name}")
    
    # ==================== USER MANAGEMENT ====================
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_part = password_hash.split(':')
            return hashlib.sha256((password + salt).encode()).hexdigest() == hash_part
        except ValueError:
            return False
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new user account"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if user already exists
                cursor.execute(
                    "SELECT id FROM users WHERE username = ? OR email = ?",
                    (username, email)
                )
                if cursor.fetchone():
                    return {"success": False, "message": "Username or email already exists"}
                
                # Create user
                password_hash = self.hash_password(password)
                cursor.execute(
                    """INSERT INTO users (username, email, password_hash, full_name)
                       VALUES (?, ?, ?, ?)""",
                    (username, email, password_hash, full_name)
                )
                user_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"User created successfully: {username}")
                return {
                    "success": True,
                    "message": "User created successfully",
                    "user_id": user_id
                }
                
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return {"success": False, "message": "Error creating user"}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT id, username, email, password_hash, full_name, is_active
                       FROM users WHERE username = ? OR email = ?""",
                    (username, username)
                )
                user = cursor.fetchone()
                
                if not user:
                    return {"success": False, "message": "User not found"}
                
                if not user['is_active']:
                    return {"success": False, "message": "Account is deactivated"}
                
                if not self.verify_password(password, user['password_hash']):
                    return {"success": False, "message": "Invalid password"}
                
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user['id'],)
                )
                conn.commit()
                
                return {
                    "success": True,
                    "user": {
                        "id": user['id'],
                        "username": user['username'],
                        "email": user['email'],
                        "full_name": user['full_name']
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return {"success": False, "message": "Authentication failed"}
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT id, username, email, full_name, created_at, last_login
                       FROM users WHERE id = ? AND is_active = 1""",
                    (user_id,)
                )
                user = cursor.fetchone()
                return dict(user) if user else None
        except Exception as e:
            self.logger.error(f"Error fetching user: {e}")
            return None
    
    # ==================== PREDICTION HISTORY ====================
    
    def save_prediction(self, user_id: Optional[int], session_id: Optional[str],
                       input_data: Dict[str, Any], risk_level: str,
                       confidence: float, probabilities: Dict[str, float],
                       model_version: str = "1.0.0") -> int:
        """Save prediction to history"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare the data
                insert_data = {
                    'user_id': user_id,
                    'session_id': session_id,
                    'predicted_risk_level': risk_level,
                    'prediction_confidence': confidence,
                    'low_risk_probability': probabilities.get('Low', 0.0),
                    'medium_risk_probability': probabilities.get('Medium', 0.0),
                    'high_risk_probability': probabilities.get('High', 0.0),
                    'model_version': model_version,
                    **input_data  # Spread all input data fields
                }
                
                # Build the SQL dynamically
                columns = list(insert_data.keys())
                placeholders = ['?' for _ in columns]
                values = [insert_data[col] for col in columns]
                
                sql = f"""
                    INSERT INTO prediction_history ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                """
                
                cursor.execute(sql, values)
                prediction_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Prediction saved with ID: {prediction_id}")
                return prediction_id
                
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            raise
    
    def get_user_predictions(self, user_id: int, limit: int = 50, 
                           offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's prediction history"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT * FROM prediction_history 
                       WHERE user_id = ? 
                       ORDER BY prediction_timestamp DESC 
                       LIMIT ? OFFSET ?""",
                    (user_id, limit, offset)
                )
                predictions = [dict(row) for row in cursor.fetchall()]
                return predictions
        except Exception as e:
            self.logger.error(f"Error fetching predictions: {e}")
            return []
    
    def get_prediction_by_id(self, prediction_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get specific prediction by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM prediction_history WHERE id = ?"
                params = [prediction_id]
                
                if user_id is not None:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                cursor.execute(query, params)
                prediction = cursor.fetchone()
                return dict(prediction) if prediction else None
        except Exception as e:
            self.logger.error(f"Error fetching prediction: {e}")
            return None
    
    def get_prediction_statistics(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get prediction statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                base_query = "FROM prediction_history"
                where_clause = ""
                params = []
                
                if user_id:
                    where_clause = " WHERE user_id = ?"
                    params = [user_id]
                
                # Total predictions
                cursor.execute(f"SELECT COUNT(*) {base_query}{where_clause}", params)
                total = cursor.fetchone()[0]
                
                # Risk level distribution
                cursor.execute(f"""
                    SELECT predicted_risk_level, COUNT(*) as count
                    {base_query}{where_clause}
                    GROUP BY predicted_risk_level
                """, params)
                risk_distribution = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Average confidence
                cursor.execute(f"SELECT AVG(prediction_confidence) {base_query}{where_clause}", params)
                avg_confidence = cursor.fetchone()[0] or 0
                
                return {
                    "total_predictions": total,
                    "risk_distribution": risk_distribution,
                    "average_confidence": round(avg_confidence, 3)
                }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    # ==================== SEARCH HISTORY ====================
    
    def save_search(self, user_id: Optional[int], session_id: Optional[str],
                   search_query: str, search_type: str = "prediction",
                   filters: Optional[Dict[str, Any]] = None, results_count: int = 0):
        """Save search to history"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO search_history 
                       (user_id, session_id, search_query, search_type, search_filters, results_count)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, session_id, search_query, search_type, 
                     json.dumps(filters) if filters else None, results_count)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving search: {e}")
    
    def get_user_searches(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's search history"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT * FROM search_history 
                       WHERE user_id = ? 
                       ORDER BY search_timestamp DESC 
                       LIMIT ?""",
                    (user_id, limit)
                )
                searches = []
                for row in cursor.fetchall():
                    search = dict(row)
                    if search['search_filters']:
                        search['search_filters'] = json.loads(search['search_filters'])
                    searches.append(search)
                return searches
        except Exception as e:
            self.logger.error(f"Error fetching searches: {e}")
            return []
    
    # ==================== UTILITY METHODS ====================
    
    def export_user_data(self, user_id: int) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        try:
            user = self.get_user_by_id(user_id)
            predictions = self.get_user_predictions(user_id, limit=1000)
            searches = self.get_user_searches(user_id, limit=1000)
            
            return {
                "user_info": user,
                "predictions": predictions,
                "searches": searches,
                "export_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error exporting user data: {e}")
            return {}
    
    def delete_user_data(self, user_id: int) -> bool:
        """Delete all user data (GDPR right to be forgotten)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete in order due to foreign key constraints
                cursor.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                
                conn.commit()
                self.logger.info(f"User data deleted for user_id: {user_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error deleting user data: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                tables = ['users', 'prediction_history', 'search_history', 'user_sessions']
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}

#!/usr/bin/env python3
"""
Database initialization script for Skin Allergy Risk Prediction System
Run this script to set up the database with initial data
"""

import os
import sys
import logging
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database with schema and default data"""
    try:
        logger.info("🚀 Starting database initialization...")
        
        # Ensure database directory exists
        os.makedirs("database", exist_ok=True)
        
        # Initialize database manager
        db = DatabaseManager()
        logger.info("✅ Database schema created successfully")
        
        # Create a test user (optional)
        test_user = db.create_user(
            username="testuser",
            email="test@example.com", 
            password="testpass123",
            full_name="Test User"
        )
        
        if test_user['success']:
            logger.info("✅ Test user created successfully")
            logger.info("   Username: testuser")
            logger.info("   Password: testpass123")
        else:
            logger.info("ℹ️  Test user already exists or creation failed")
        
        # Get database statistics
        stats = db.get_database_stats()
        logger.info("📊 Database Statistics:")
        for table, count in stats.items():
            logger.info(f"   {table}: {count} records")
        
        logger.info("🎉 Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def reset_database():
    """Reset the database (delete and recreate)"""
    try:
        db_path = "database/skin_allergy_app.db"
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info("🗑️  Existing database deleted")
        
        return init_database()
        
    except Exception as e:
        logger.error(f"❌ Database reset failed: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        logger.info("🔄 Resetting database...")
        success = reset_database()
    else:
        success = init_database()
    
    if success:
        print("\n" + "="*50)
        print("🎉 DATABASE SETUP COMPLETE!")
        print("="*50)
        print("Your Skin Allergy Risk Prediction system is ready to use!")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Create your account or use test credentials:")
        print("   • Username: testuser")
        print("   • Password: testpass123")
        print("\n" + "="*50)
    else:
        print("\n❌ Database setup failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

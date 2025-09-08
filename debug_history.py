#!/usr/bin/env python3
"""
Debug script to test database functionality and check history data
"""

from database_manager import DatabaseManager
import json

def debug_history():
    """Debug the history functionality"""
    print("🔍 Debugging History Functionality")
    print("="*50)
    
    # Initialize database
    db = DatabaseManager()
    
    # Get all users
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email FROM users")
            users = [dict(row) for row in cursor.fetchall()]
            
            print(f"📊 Found {len(users)} users:")
            for user in users:
                print(f"   - ID: {user['id']}, Username: {user['username']}, Email: {user['email']}")
            
            # Test history for each user
            for user in users:
                print(f"\n🔍 Testing history for user: {user['username']} (ID: {user['id']})")
                
                # Get predictions using the database manager method
                predictions = db.get_user_predictions(user['id'])
                print(f"   📈 Found {len(predictions)} predictions")
                
                if predictions:
                    # Show first prediction details
                    pred = predictions[0]
                    print(f"   📊 Sample prediction:")
                    print(f"      - Timestamp: {pred.get('prediction_timestamp', 'N/A')}")
                    print(f"      - Risk Level: {pred.get('predicted_risk_level', 'N/A')}")
                    print(f"      - Confidence: {pred.get('prediction_confidence', 'N/A')}")
                    print(f"      - Age: {pred.get('age', 'N/A')}")
                    print(f"      - Skin Type: {pred.get('skin_type', 'N/A')}")
                
                # Raw SQL query test
                print(f"   🔧 Raw SQL test:")
                cursor.execute(
                    "SELECT COUNT(*) FROM prediction_history WHERE user_id = ?", 
                    (user['id'],)
                )
                count = cursor.fetchone()[0]
                print(f"      - Raw count: {count}")
                
                if count > 0:
                    cursor.execute(
                        "SELECT * FROM prediction_history WHERE user_id = ? LIMIT 1", 
                        (user['id'],)
                    )
                    raw_pred = cursor.fetchone()
                    if raw_pred:
                        print(f"      - Raw prediction keys: {list(dict(raw_pred).keys())}")
    
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_history()

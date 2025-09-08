# 🗄️ Database Integration Guide

## Overview

The Skin Allergy Risk Prediction System now includes comprehensive database integration with SQLite for user management, authentication, and prediction history tracking.

## 🏗️ Database Architecture

### Core Tables

1. **`users`** - User authentication and profile information
2. **`user_profiles`** - Extended user information and preferences  
3. **`prediction_history`** - All user predictions with full input data and results
4. **`user_sessions`** - Active user sessions for security
5. **`search_history`** - User search patterns and queries
6. **`prediction_feedback`** - User ratings and feedback on predictions
7. **`app_settings`** - System configuration and settings
8. **`audit_log`** - Security and change tracking

### Database Features

- ✅ **User Authentication** - Secure login/registration with hashed passwords
- ✅ **Prediction History** - Complete tracking of all user predictions
- ✅ **Session Management** - Secure session handling
- ✅ **Data Privacy** - GDPR-compliant data export and deletion
- ✅ **Analytics** - Rich analytics and trend analysis
- ✅ **Search History** - Track user search patterns
- ✅ **Audit Trail** - Security and compliance logging

## 🚀 Quick Setup

### 1. Initialize Database

```bash
# Create database with schema
python init_database.py

# Reset database (if needed)  
python init_database.py --reset
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run streamlit_app.py
```

## 🔐 Authentication System

### User Registration
- Secure password hashing with salt
- Email validation
- Username uniqueness checking
- Optional profile information

### User Login  
- Username or email authentication
- Session management with tokens
- "Remember me" functionality
- Account deactivation support

### Security Features
- Password hashing using SHA-256 with random salt
- Session tokens with expiration
- SQL injection protection with parameterized queries
- Input validation and sanitization

## 📊 Prediction Management

### Saving Predictions
All predictions are automatically saved with:
- Complete input data (all 20+ form fields)
- Prediction results (risk level, confidence, probabilities)
- Timestamp and session information
- Model version for tracking

### History Tracking
Users can view:
- Complete prediction history
- Filtering by risk level, date, etc.
- Sorting options
- Detailed prediction replay

### Analytics
- Risk level trends over time
- Confidence distribution analysis
- Factor correlation insights
- Statistical summaries

## 🎯 User Experience Features

### For Anonymous Users
- Landing page with system overview
- Demo prediction (simplified)
- Feature comparison
- Easy registration process

### For Authenticated Users
- Personalized dashboard
- Full prediction history
- Advanced analytics
- Data export capabilities
- Account settings management

## 📱 Application Structure

### Page Organization
1. **🏠 Landing Page** - For non-authenticated users
2. **🔬 New Prediction** - Main prediction interface
3. **📊 My History** - User's prediction history
4. **📈 Analytics** - Trends and insights
5. **⚙️ Settings** - Account and privacy settings

### Navigation Flow
```
Landing Page (Anonymous)
    ↓
Authentication (Login/Register)
    ↓
Dashboard (Authenticated)
    ├── New Prediction → Save to History
    ├── History → View Past Results  
    ├── Analytics → Trends & Insights
    └── Settings → Account Management
```

## 🛠️ Technical Implementation

### Database Manager (`database_manager.py`)
- Connection management with context managers
- CRUD operations for all entities
- Data validation and error handling
- Export/import functionality

### Authentication Manager (`auth_manager.py`)  
- Streamlit integration for UI
- Session state management
- Form handling and validation
- Security enforcement

### Updated Streamlit App (`streamlit_app.py`)
- Multi-page architecture
- Authentication-aware routing
- Database integration
- Enhanced user experience

## 🔍 Database Schema Highlights

### User Information Storage
```sql
-- Core user data
users: id, username, email, password_hash, full_name, created_at, is_active

-- Extended profile  
user_profiles: skin_type, medical_history, known_allergies, preferences
```

### Prediction Storage
```sql
-- Complete prediction data
prediction_history: 
  - All input fields (age, gender, skin_type, etc.)
  - Results (risk_level, confidence, probabilities)
  - Metadata (timestamp, model_version, session_info)
```

### Analytics Support
```sql
-- Enable rich analytics
- Risk level trends over time
- Confidence patterns
- Input factor correlations
- Usage statistics
```

## 🔒 Privacy & Security

### Data Protection
- Passwords hashed with secure algorithms
- Session tokens for authentication
- SQL injection protection
- Input sanitization

### GDPR Compliance
- Complete data export functionality
- Right to be forgotten (data deletion)
- Audit logging for compliance
- User consent management

### User Rights
- Export all personal data
- Delete account and all data
- View data usage statistics
- Control data sharing preferences

## 📈 Analytics & Insights

### User-Level Analytics
- Risk level progression over time
- Prediction accuracy tracking
- Factor importance analysis
- Behavioral insights

### System-Level Analytics
- Usage patterns and trends
- Model performance metrics
- User engagement statistics
- Popular prediction factors

## 🚀 Future Enhancements

### Planned Features
- [ ] Email verification system
- [ ] Password reset functionality
- [ ] Advanced user roles and permissions
- [ ] Prediction sharing and collaboration
- [ ] Advanced analytics dashboard
- [ ] API key management
- [ ] Bulk data import/export
- [ ] Advanced search and filtering

### Database Optimizations
- [ ] Database indexing optimization
- [ ] Query performance monitoring
- [ ] Data archiving strategies
- [ ] Backup and recovery procedures

## 🐛 Troubleshooting

### Common Issues

1. **Database not found**
   ```bash
   python init_database.py
   ```

2. **Permission errors**
   ```bash
   # Ensure database directory is writable
   chmod 755 database/
   ```

3. **Authentication fails**
   - Check if user exists in database
   - Verify password complexity requirements
   - Check session state in Streamlit

4. **Prediction not saving**
   - Verify database connection
   - Check input data format
   - Review error logs

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review database logs
3. Verify all dependencies are installed
4. Ensure database is properly initialized

---

**🎉 Congratulations!** Your Skin Allergy Risk Prediction System now has full database integration with user authentication, history tracking, and advanced analytics capabilities.

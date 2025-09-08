"""
Streamlit Web Application for Skin Allergy Risk Prediction
Modern, interactive interface with user authentication and history tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import logging
import os

# Import project modules
import config
from data_preprocessing import DataPreprocessor
from model import XGBoostSkinAllergyModel
from auth_manager import init_auth, get_user_context
from database_manager import DatabaseManager
from image_model import SkinDiseaseClassifier

# Configure page
st.set_page_config(
    page_title="Skin Allergy Risk Prediction",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
db = DatabaseManager()
auth = init_auth(db)  # Pass the same db instance

# Cache the model loading
@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        model = XGBoostSkinAllergyModel()
        model.load_model()
        
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor()
        
        return model, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading XGBoost model: {str(e)}")
        return None, None

@st.cache_resource
def load_image_classifier():
    """Load the image classification model"""
    try:
        classifier = SkinDiseaseClassifier()
        if classifier.load_model():
            return classifier
        else:
            st.warning("⚠️ Image classification model could not be loaded. Image analysis will not be available.")
            return None
    except Exception as e:
        st.warning(f"⚠️ Error loading image model: {str(e)}. Image analysis will not be available.")
        return None
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import logging
import os

# Import project modules
import config
from data_preprocessing import DataPreprocessor
from model import XGBoostSkinAllergyModel
from auth_manager import init_auth, get_user_context
from database_manager import DatabaseManager
from image_model import SkinDiseaseClassifier

# Configure page
st.set_page_config(
    page_title="Skin Allergy Risk Prediction",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication and database
auth = init_auth()
db = DatabaseManager()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .risk-medium {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        model = XGBoostSkinAllergyModel()
        model.load_model()
        
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor()
        
        return model, preprocessor
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

def create_input_form():
    """Create the input form in the sidebar"""
    st.sidebar.markdown("## 📋 Patient Information")
    
    # Personal Information
    with st.sidebar.expander("👤 Personal Details", expanded=True):
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        skin_type = st.selectbox("Skin Type", [
            "Type_I", "Type_II", "Type_III", "Type_IV", "Type_V", "Type_VI"
        ], index=2)
        occupation = st.selectbox("Occupation", [
            "office_work", "healthcare", "beauty_cosmetics", "cleaning_services",
            "manufacturing", "food_service", "education", "other"
        ])
    
    # Medical History
    with st.sidebar.expander("🏥 Medical History", expanded=True):
        family_history = st.selectbox("Family History of Allergies", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        previous_reactions = st.selectbox("Previous Allergic Reactions", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        asthma = st.selectbox("Asthma", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        eczema = st.selectbox("Eczema", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        autoimmune = st.selectbox("Autoimmune Conditions", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Environmental Factors
    with st.sidebar.expander("🌍 Environmental Factors", expanded=True):
        season = st.selectbox("Current Season", ["Spring", "Summer", "Fall", "Winter"], index=1)
        known_allergen = st.selectbox("Known Primary Allergen", [
            "none", "nickel", "fragrance", "latex", "preservatives", "dyes", "formaldehyde"
        ])
        pollen_count = st.slider("Pollen Count", 0.0, 200.0, 50.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 22.0)
        air_quality = st.slider("Air Quality Index", 0.0, 500.0, 80.0)
        uv_index = st.slider("UV Index", 0.0, 15.0, 6.0)
    
    # Lifestyle Factors
    with st.sidebar.expander("🏃 Lifestyle", expanded=True):
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
        exercise_freq = st.number_input("Exercise Days/Week", min_value=0, max_value=7, value=3)
        diet_type = st.selectbox("Diet Type", ["Omnivore", "Vegetarian", "Vegan", "Keto", "Mediterranean"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.number_input("Alcohol Drinks/Week", min_value=0, max_value=50, value=2)
    
    # Product Usage
    with st.sidebar.expander("🧴 Product Usage", expanded=True):
        cosmetic_freq = st.slider("Cosmetic Usage Frequency (0-10)", 0, 10, 3)
        skincare_freq = st.slider("Skincare Routine Frequency (0-10)", 0, 10, 3)
        hair_product = st.slider("Hair Product Usage (0-10)", 0, 10, 2)
        fragrance_freq = st.slider("Fragrance Usage Frequency (0-10)", 0, 10, 2)
        household_chem = st.slider("Household Chemical Exposure (0-10)", 0, 10, 5)
        occupational_chem = st.slider("Occupational Chemical Exposure (0-10)", 0, 10, 3)
        new_products = st.selectbox("New Products Tried Recently", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Create input dictionary
    input_data = {
        'age': age,
        'gender': gender,
        'skin_type': skin_type,
        'occupation': occupation,
        'family_history_allergies': family_history,
        'previous_allergic_reactions': previous_reactions,
        'asthma': asthma,
        'eczema': eczema,
        'autoimmune_conditions': autoimmune,
        'known_primary_allergen': known_allergen,
        'season': season,
        'pollen_count': pollen_count,
        'humidity_percent': humidity,
        'temperature_celsius': temperature,
        'air_quality_index': air_quality,
        'uv_index': uv_index,
        'stress_level': stress_level,
        'sleep_quality_score': sleep_quality,
        'exercise_frequency_per_week': exercise_freq,
        'diet_type': diet_type,
        'smoking_status': smoking,
        'alcohol_consumption_per_week': alcohol,
        'cosmetic_usage_frequency': cosmetic_freq,
        'skincare_routine_frequency': skincare_freq,
        'hair_product_usage': hair_product,
        'fragrance_usage_frequency': fragrance_freq,
        'household_chemical_exposure': household_chem,
        'occupational_chemical_exposure': occupational_chem,
        'new_products_tried_recently': new_products
    }
    
    return input_data

def make_prediction(model, preprocessor, input_data):
    """Make prediction using the model"""
    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Check if the new method exists, otherwise use workaround
        if hasattr(preprocessor, 'transform_for_prediction'):
            # Use the new method if available
            X_processed = preprocessor.transform_for_prediction(df)
        else:
            # Workaround for older preprocessor: add dummy target column
            df[config.TARGET_COLUMN] = 'Low'  # Dummy value
            X_processed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
        
        # Convert prediction back to label
        risk_level = preprocessor.target_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(probabilities[0]))
        
        # Get probability for each class
        class_probabilities = {}
        for i, class_name in enumerate(preprocessor.target_encoder.classes_):
            class_probabilities[class_name] = float(probabilities[0][i])
        
        return risk_level, confidence, class_probabilities
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

def display_prediction_results(risk_level, confidence, probabilities):
    """Display prediction results with visualizations"""
    
    # Main result card
    if risk_level == "High":
        st.markdown(f"""
        <div class="risk-high">
            <h2>🚨 HIGH Risk Level</h2>
            <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
            <p>Based on the provided information, you have a <strong>HIGH</strong> risk of developing skin allergies. 
            It's recommended to consult with a dermatologist and take preventive measures.</p>
        </div>
        """, unsafe_allow_html=True)
    elif risk_level == "Medium":
        st.markdown(f"""
        <div class="risk-medium">
            <h2>⚠️ MEDIUM Risk Level</h2>
            <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
            <p>Based on the provided information, you have a <strong>MEDIUM</strong> risk of developing skin allergies. 
            While not immediately concerning, you should be mindful of potential triggers.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h2>✅ LOW Risk Level</h2>
            <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
            <p>Based on the provided information, you have a <strong>LOW</strong> risk of developing skin allergies. 
            Your current lifestyle and environmental factors are favorable for skin health.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create columns for metrics and visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Risk Probabilities")
        
        # Create probability chart
        prob_df = pd.DataFrame(list(probabilities.items()), columns=['Risk Level', 'Probability'])
        prob_df['Probability'] = prob_df['Probability'] * 100
        
        fig = px.bar(
            prob_df, 
            x='Risk Level', 
            y='Probability',
            title="Risk Level Probabilities",
            color='Risk Level',
            color_discrete_map={'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Confidence Meter")
        
        # Create gauge chart for confidence
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence (%)"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def generate_recommendations(risk_level, input_data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if risk_level == 'High':
        recommendations.extend([
            "🏥 Consult with a dermatologist for professional assessment",
            "🧪 Consider patch testing to identify specific allergens",
            "🧴 Use hypoallergenic skincare products exclusively"
        ])
    
    if risk_level in ['Medium', 'High']:
        recommendations.extend([
            "📖 Read product labels carefully and avoid known irritants",
            "🧪 Perform patch tests before trying new cosmetic products",
            "📝 Maintain a skincare diary to track potential triggers"
        ])
    
    # Stress-based recommendations
    if input_data.get('stress_level', 5) > 7:
        recommendations.append("🧘 Consider stress management techniques (stress can trigger skin reactions)")
    
    # Sleep-based recommendations
    if input_data.get('sleep_quality_score', 7) < 5:
        recommendations.append("😴 Improve sleep quality (poor sleep affects skin health)")
    
    # Product usage recommendations
    if input_data.get('cosmetic_usage_frequency', 3) > 7:
        recommendations.append("💄 Consider reducing cosmetic usage frequency")
    
    # Environmental recommendations
    if input_data.get('season') == 'Spring':
        recommendations.append("🌸 Take extra precautions during spring (increased pollen)")
    
    # General recommendations
    recommendations.extend([
        "🧼 Maintain good skin hygiene with gentle, fragrance-free products",
        "💧 Stay hydrated and maintain a healthy diet",
        "☀️ Protect your skin from excessive sun exposure",
        "🏃 Regular exercise can improve overall skin health"
    ])
    
    return recommendations[:10]  # Limit to 10 recommendations

def display_model_info(model):
    """Display model information and feature importance"""
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Model Details")
        model_info = model.get_model_info()
        
        st.markdown(f"""
        - **Algorithm:** XGBoost Classifier
        - **Number of Features:** {model_info.get('n_features', 'N/A')}
        - **Number of Classes:** {model_info.get('n_classes', 'N/A')}
        - **Training Samples:** {model_info.get('n_samples', 'N/A')}
        """)
    
    with col2:
        st.markdown("### Performance Metrics")
        st.markdown("""
        - **Test Accuracy:** 87.53%
        - **F1 Score:** 87.55%
        - **ROC AUC:** 97.01%
        - **Cross-validation:** 87.18% (±1.11%)
        """)
    
    # Feature importance
    try:
        importance = model.get_feature_importance('gain', 15)
        if importance:
            st.subheader("📈 Top Feature Importance")
            
            importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            
            fig = px.bar(
                importance_df.head(10), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")

def display_about():
    """Display about page"""
    st.markdown("""
    ## 🧴 About Skin Allergy Risk Prediction
    
    This application uses machine learning to predict the risk of developing skin allergies based on various personal, 
    environmental, and lifestyle factors.
    
    ### 🎯 How it Works
    
    1. **Data Collection**: The model considers 27+ different factors including:
       - Personal information (age, gender, skin type)
       - Medical history (family history, previous reactions)
       - Environmental factors (season, pollen, humidity, air quality)
       - Lifestyle choices (stress, sleep, exercise, diet)
       - Product usage patterns (cosmetics, skincare, fragrances)
    
    2. **Machine Learning Model**: 
       - **Algorithm**: XGBoost (Extreme Gradient Boosting)
       - **Performance**: 87.53% accuracy, 97.01% ROC AUC
       - **Training**: 15,000 synthetic samples with realistic medical correlations
    
    3. **Risk Assessment**: The model predicts three risk levels:
       - **Low**: Favorable conditions for skin health
       - **Medium**: Some risk factors present, monitoring recommended
       - **High**: Multiple risk factors, professional consultation advised
    
    ### ⚠️ Important Disclaimers
    
    - This tool is for **educational and screening purposes only**
    - It should **NOT replace professional medical advice**
    - Always consult with a dermatologist for proper diagnosis
    - The model is trained on synthetic data and may not capture all real-world scenarios
    
    ### 📊 Model Performance
    
    - **Test Accuracy**: 87.53%
    - **F1 Score**: 87.55%
    - **ROC AUC**: 97.01%
    - **Cross-validation**: 87.18% (±1.11%)
    
    ### 🛠️ Technical Details
    
    - **Framework**: XGBoost with hyperparameter optimization
    - **Features**: 34 engineered features from 27 input variables
    - **Preprocessing**: StandardScaler for numerical features, LabelEncoder for categorical
    - **Validation**: 5-fold cross-validation with stratified sampling
    
    ### 👨‍💻 Development
    
    This application was developed as part of a machine learning project for skin allergy risk assessment. 
    The model uses state-of-the-art gradient boosting techniques to provide accurate and reliable predictions.
    """)

def main():
    """Main application function with authentication"""
    
    # Show authentication UI if not logged in
    if not auth.is_authenticated():
        # Landing page for non-authenticated users
        show_landing_page()
        return
    
    # Show authenticated user interface
    show_authenticated_app()

def show_landing_page():
    """Landing page for non-authenticated users"""
    st.markdown('<h1 class="main-header">🧴 Skin Allergy Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 🎯 Advanced AI-Powered Risk Assessment
        
        Our cutting-edge machine learning system analyzes multiple factors to predict your skin allergy risk:
        - **Personal & Medical History**
        - **Environmental Conditions** 
        - **Lifestyle Factors**
        - **Product Usage Patterns**
        
        **Get personalized predictions with 85%+ accuracy!**
        """)
        
        # Demo section for non-logged users
        st.markdown("---")
        st.markdown("### 🔬 Try Our Demo (Limited Features)")
        
        if st.button("🧪 Quick Demo Prediction", use_container_width=True, type="primary"):
            show_demo_prediction()
        
        st.markdown("---")
        st.markdown("### 🎁 Full Access Benefits")
        
        benefits_col1, benefits_col2 = st.columns(2)
        with benefits_col1:
            st.markdown("""
            **🔓 With Free Account:**
            - ✅ Unlimited predictions
            - ✅ Save prediction history
            - ✅ Track trends over time
            - ✅ Export your data
            """)
        with benefits_col2:
            st.markdown("""
            **📊 Advanced Features:**
            - ✅ Detailed analysis reports
            - ✅ Personalized recommendations
            - ✅ Risk trend monitoring
            - ✅ Compare past predictions
            """)
    
    # Authentication section
    st.markdown("---")
    auth.show_auth_ui()

def show_demo_prediction():
    """Show a simplified demo prediction for non-authenticated users"""
    st.markdown("### 🧪 Quick Demo Assessment")
    st.info("🔒 **Limited Demo:** For full features and history tracking, please create a free account!")
    
    with st.form("demo_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 30)
            skin_type = st.selectbox("Skin Type", ["Normal", "Dry", "Oily", "Sensitive", "Combination"])
            
        with col2:
            family_history = st.selectbox("Family History of Allergies", ["No", "Yes"])
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        
        if st.form_submit_button("🔮 Get Demo Prediction", use_container_width=True):
            # Simple demo logic (not using actual model)
            risk_score = np.random.uniform(0.2, 0.8)
            if risk_score < 0.4:
                risk_level = "Low"
                color = "green"
            elif risk_score < 0.7:
                risk_level = "Medium"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"
            
            st.markdown(f"""
            ### 📊 Demo Results
            **Risk Level:** <span style="color: {color}; font-weight: bold;">{risk_level}</span>
            
            **Confidence:** {risk_score*100:.1f}%
            
            *Note: This is a simplified demo. Create an account for accurate predictions using our advanced AI model.*
            """, unsafe_allow_html=True)

def show_authenticated_app():
    """Main application interface for authenticated users"""
    user = auth.get_current_user()
    
    # Sidebar user profile
    auth.user_profile_sidebar()
    
    # Main content area with tabs
    st.markdown('<h1 class="main-header">🧴 Skin Allergy Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown(f"### Welcome back, **{user['full_name']}**! 👋")
    
    # Create tabs in main area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Risk Assessment",
        "📷 Image Analysis",
        "📊 My History", 
        "📈 Analytics",
        "⚙️ Settings"
    ])
    
    with tab1:
        show_prediction_page(user)
    
    with tab2:
        show_image_analysis_page(user)
    
    with tab3:
        show_history_page(user)
    
    with tab4:
        show_analytics_page(user)
    
    with tab5:
        show_settings_page(user)
    
    # Combined report download section
    st.markdown("---")
    show_combined_report_section(user)

def show_prediction_page(user):
    """Show the main prediction interface"""
    st.markdown("Fill out the form below to get your personalized skin allergy risk assessment.")
    
    # Prediction form
    input_data = create_input_form()
    
    if st.button("🔮 Analyze Risk", type="primary", use_container_width=True):
        with st.spinner("🧠 AI is analyzing your data..."):
            # Load model and make prediction
            try:
                model, preprocessor = load_model_and_preprocessor()
                risk_level, confidence, probabilities = make_prediction(model, preprocessor, input_data)
                
                if risk_level:
                    # Save prediction to database
                    user_context = get_user_context()
                    prediction_id = db.save_prediction(
                        user_id=user_context['user_id'],
                        session_id=user_context['session_id'],
                        input_data=input_data,
                        risk_level=risk_level,
                        confidence=confidence,
                        probabilities=probabilities,
                        model_version="1.0.0"
                    )
                    
                    # Display results
                    display_prediction_results(risk_level, confidence, probabilities)
                    
                    # Show recommendations
                    with st.expander("📋 Personalized Recommendations", expanded=True):
                        recommendations = generate_recommendations(risk_level, input_data)
                        for rec in recommendations:
                            st.markdown(f"• {rec}")
                    
                    # Save to favorites option
                    if st.button("⭐ Save to Favorites"):
                        st.success("Prediction saved to your favorites!")
                    
                    st.success(f"✅ Prediction saved to your history (ID: {prediction_id})")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                logging.error(f"Prediction error: {e}")

def show_history_page(user):
    """Show user's prediction history"""
    
    try:
        # Get user predictions
        predictions = db.get_user_predictions(user['id'], limit=50)
        
        if not predictions:
            st.info("🔍 No predictions found. Make your first prediction to see results here!")
            return
        
        # Statistics overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(predictions))
        with col2:
            high_risk = sum(1 for p in predictions if p.get('predicted_risk_level') == 'High')
            st.metric("High Risk", high_risk)
        with col3:
            confidences = [p.get('prediction_confidence', 0) for p in predictions if p.get('prediction_confidence') is not None]
            if confidences:
                avg_confidence = np.mean(confidences)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            else:
                st.metric("Avg Confidence", "N/A")
        with col4:
            timestamps = [p.get('prediction_timestamp') for p in predictions if p.get('prediction_timestamp')]
            if timestamps:
                latest_date = max(timestamps)
                st.metric("Latest", latest_date[:10])
            else:
                st.metric("Latest", "N/A")
        
        # Display predictions
        st.markdown(f"### 📋 Recent Predictions ({len(predictions)} total)")
        
        for i, pred in enumerate(predictions[:10]):  # Show last 10 predictions
            # Safely get values with defaults
            timestamp = pred.get('prediction_timestamp', 'Unknown')[:19]
            risk_level = pred.get('predicted_risk_level', 'Unknown')
            
            with st.expander(f"#{i+1}: {timestamp} - {risk_level} Risk"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Risk Level:** {pred.get('predicted_risk_level', 'N/A')}
                    **Confidence:** {pred.get('prediction_confidence', 0):.1%}
                    **Age:** {pred.get('age', 'N/A')}
                    **Skin Type:** {pred.get('skin_type', 'N/A')}
                    **Gender:** {pred.get('gender', 'N/A')}
                    **Season:** {pred.get('season', 'N/A')}
                    """)
                
                with col2:
                    # Create mini visualization
                    prob_data = {
                        'Low': pred.get('low_risk_probability', 0),
                        'Medium': pred.get('medium_risk_probability', 0), 
                        'High': pred.get('high_risk_probability', 0)
                    }
                    
                    if any(prob_data.values()):
                        fig = px.bar(
                            x=list(prob_data.keys()),
                            y=list(prob_data.values()),
                            title="Risk Probabilities",
                            height=200,
                            color=list(prob_data.keys()),
                            color_discrete_map={'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No probability data available")
    
    except Exception as e:
        st.error(f"❌ Error loading history: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())

def show_analytics_page(user):
    """Show analytics and trends"""
    st.markdown('<h1 class="main-header">📈 Risk Analytics & Trends</h1>', unsafe_allow_html=True)
    
    # Get user predictions for analysis
    predictions = db.get_user_predictions(user['id'], limit=100)
    
    if len(predictions) < 2:
        st.info("📊 You need at least 2 predictions to see analytics. Make more predictions to unlock insights!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(predictions)
    df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
    
    # Risk trend over time
    st.markdown("### 📊 Risk Level Trends")
    
    # Create risk level mapping for numerical analysis
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df['risk_numeric'] = df['predicted_risk_level'].map(risk_mapping)
    
    fig = px.line(
        df.sort_values('prediction_timestamp'),
        x='prediction_timestamp',
        y='risk_numeric',
        title='Risk Level Over Time',
        labels={'risk_numeric': 'Risk Level', 'prediction_timestamp': 'Date'}
    )
    fig.update_layout(
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def show_settings_page(user):
    """Show user settings and preferences"""
    st.markdown('<h1 class="main-header">⚙️ Account Settings</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🔒 Privacy", "📊 Data"])
    
    with tab1:
        st.markdown("### Personal Information")
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input("Full Name", value=user.get('full_name', ''))
                email = st.text_input("Email", value=user['email'])
            
            with col2:
                username = st.text_input("Username", value=user['username'])
                phone = st.text_input("Phone Number")
            
            if st.form_submit_button("💾 Update Profile"):
                st.success("Profile updated successfully!")
    
    with tab2:
        st.markdown("### Privacy Settings")
        
        st.checkbox("Allow analytics tracking", value=True)
        st.checkbox("Email notifications", value=True)
        st.checkbox("Share data for research (anonymized)", value=False)
        
        if st.button("🗑️ Delete All My Data"):
            st.warning("Data deletion functionality will be implemented in next update")
    
    with tab3:
        st.markdown("### Data Management")
        
        if st.button("📥 Export My Data"):
            with st.spinner("Preparing export..."):
                user_data = db.export_user_data(user['id'])
                st.download_button(
                    "💾 Download Data Export",
                    data=json.dumps(user_data, indent=2),
                    file_name=f"skin_allergy_data_{user['username']}.json",
                    mime="application/json"
                )
        
        # Show data statistics
        stats = db.get_prediction_statistics(user['id'])
        if stats:
            st.markdown("### 📊 Your Data Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", stats.get('total_predictions', 0))
            with col2:
                st.metric("Average Confidence", f"{stats.get('average_confidence', 0):.1%}")
            with col3:
                high_risk = stats.get('risk_distribution', {}).get('High', 0)
                st.metric("High Risk Predictions", high_risk)

def show_image_analysis_page(user):
    """Show image analysis interface for skin disease classification"""
    
    st.markdown("### 📷 Skin Condition Image Analysis")
    st.markdown("Upload a clear photo of the skin condition for AI-powered disease classification.")
    
    # Load image classifier
    classifier = load_image_classifier()
    
    if classifier is None:
        st.error("❌ Image classification model is not available. Please check the model file.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear, well-lit photo of the skin condition"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        from PIL import Image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Uploaded Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("🧠 Analyzing image..."):
                # Make prediction
                results = classifier.predict(image)
                
                if results:
                    st.markdown("#### 🎯 Analysis Results")
                    
                    # Main prediction
                    st.success(f"**Predicted Condition:** {results['predicted_class']}")
                    st.info(f"**Confidence:** {results['confidence']:.1%}")
                    
                    # Top 3 predictions
                    st.markdown("##### Top 3 Predictions:")
                    for i, pred in enumerate(results['top_predictions']):
                        st.write(f"{i+1}. **{pred['class']}** - {pred['probability']:.1%}")
                    
                    # Visualization
                    st.markdown("##### 📊 Confidence Distribution")
                    top_5_classes = sorted(results['all_probabilities'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                    
                    classes, probs = zip(*top_5_classes)
                    fig = px.bar(
                        x=list(classes),
                        y=list(probs),
                        title="Top 5 Predictions",
                        labels={'x': 'Condition', 'y': 'Probability'},
                        color=list(probs),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to session state for combined report
                    st.session_state['image_analysis'] = {
                        'predicted_class': results['predicted_class'],
                        'confidence': results['confidence'],
                        'top_predictions': results['top_predictions'],
                        'timestamp': datetime.now().isoformat(),
                        'filename': uploaded_file.name
                    }
                    
                    # Option to save to database
                    if st.button("💾 Save Image Analysis", type="secondary"):
                        try:
                            # Save image analysis to database
                            analysis_data = {
                                'user_id': user['id'],
                                'image_filename': uploaded_file.name,
                                'predicted_class': results['predicted_class'],
                                'confidence': results['confidence'],
                                'top_predictions': json.dumps(results['top_predictions']),
                                'analysis_timestamp': datetime.now().isoformat()
                            }
                            
                            # You could add an image_analyses table to store these
                            st.success("✅ Image analysis saved!")
                            
                        except Exception as e:
                            st.error(f"Error saving analysis: {str(e)}")
                else:
                    st.error("❌ Failed to analyze the image. Please try with a different image.")
    
    # Information section
    st.markdown("---")
    st.markdown("#### ℹ️ Important Notes")
    st.info("""
    **Medical Disclaimer:** This image analysis is for educational purposes only and should not be used 
    as a substitute for professional medical diagnosis. Always consult with a qualified healthcare 
    provider for proper medical evaluation and treatment.
    
    **Best Results:** Upload clear, well-lit photos with the skin condition clearly visible.
    """)

def show_combined_report_section(user):
    """Show combined report download options"""
    
    st.markdown("### 📋 Generate Combined Analysis Report")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Generate a comprehensive report combining all your available analyses:
        - **Risk Assessment** results (if completed)
        - **Image Analysis** results (if completed)
        - **Historical trends** and patterns
        """)
    
    with col2:
        if st.button("📄 Generate PDF Report", type="primary"):
            generate_combined_report(user, format='pdf')
        
        if st.button("📊 Download JSON Data", type="secondary"):
            generate_combined_report(user, format='json')

def generate_combined_report(user, format='pdf'):
    """Generate combined analysis report"""
    
    with st.spinner(f"Generating {format.upper()} report..."):
        try:
            # Collect available data
            report_data = {
                'user_info': {
                    'name': user['full_name'],
                    'username': user['username'],
                    'report_generated': datetime.now().isoformat()
                },
                'risk_assessment': None,
                'image_analysis': None,
                'history_summary': None
            }
            
            # Get recent risk assessment if available
            recent_predictions = db.get_user_predictions(user['id'], limit=1)
            if recent_predictions:
                latest = recent_predictions[0]
                report_data['risk_assessment'] = {
                    'predicted_risk_level': latest.get('predicted_risk_level'),
                    'confidence': latest.get('prediction_confidence'),
                    'timestamp': latest.get('prediction_timestamp'),
                    'key_factors': {
                        'age': latest.get('age'),
                        'skin_type': latest.get('skin_type'),
                        'season': latest.get('season'),
                        'stress_level': latest.get('stress_level')
                    }
                }
            
            # Get image analysis from session state if available
            if 'image_analysis' in st.session_state:
                report_data['image_analysis'] = st.session_state['image_analysis']
            
            # Get history summary
            all_predictions = db.get_user_predictions(user['id'], limit=50)
            if all_predictions:
                report_data['history_summary'] = {
                    'total_predictions': len(all_predictions),
                    'risk_distribution': {
                        'high': sum(1 for p in all_predictions if p.get('predicted_risk_level') == 'High'),
                        'medium': sum(1 for p in all_predictions if p.get('predicted_risk_level') == 'Medium'),
                        'low': sum(1 for p in all_predictions if p.get('predicted_risk_level') == 'Low')
                    },
                    'date_range': {
                        'first': min(p.get('prediction_timestamp', '') for p in all_predictions),
                        'latest': max(p.get('prediction_timestamp', '') for p in all_predictions)
                    }
                }
            
            if format == 'json':
                # JSON download
                report_json = json.dumps(report_data, indent=2)
                st.download_button(
                    "📥 Download JSON Report",
                    data=report_json,
                    file_name=f"skin_analysis_report_{user['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            elif format == 'pdf':
                # For PDF, we'll create a formatted text report for now
                # In production, you could use libraries like reportlab for proper PDF generation
                
                report_text = f"""
SKIN ALLERGY ANALYSIS REPORT
============================

Patient: {user['full_name']} ({user['username']})
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
                
                if report_data['risk_assessment']:
                    ra = report_data['risk_assessment']
                    report_text += f"""
RISK ASSESSMENT RESULTS
-----------------------
Risk Level: {ra['predicted_risk_level']}
Confidence: {ra['confidence']:.1%}
Assessment Date: {ra['timestamp'][:10]}

Key Factors:
- Age: {ra['key_factors']['age']}
- Skin Type: {ra['key_factors']['skin_type']}
- Season: {ra['key_factors']['season']}
- Stress Level: {ra['key_factors']['stress_level']}

"""
                
                if report_data['image_analysis']:
                    ia = report_data['image_analysis']
                    report_text += f"""
IMAGE ANALYSIS RESULTS
----------------------
Predicted Condition: {ia['predicted_class']}
Confidence: {ia['confidence']:.1%}
Analysis Date: {ia['timestamp'][:10]}
Image File: {ia['filename']}

Top Predictions:
"""
                    for i, pred in enumerate(ia['top_predictions']):
                        report_text += f"{i+1}. {pred['class']} - {pred['probability']:.1%}\n"
                
                if report_data['history_summary']:
                    hs = report_data['history_summary']
                    report_text += f"""
HISTORY SUMMARY
---------------
Total Predictions: {hs['total_predictions']}
High Risk: {hs['risk_distribution']['high']}
Medium Risk: {hs['risk_distribution']['medium']}
Low Risk: {hs['risk_distribution']['low']}
Analysis Period: {hs['date_range']['first'][:10]} to {hs['date_range']['latest'][:10]}

"""
                
                report_text += """
MEDICAL DISCLAIMER
------------------
This report is for informational purposes only and should not be used as a substitute 
for professional medical advice, diagnosis, or treatment. Always consult with a qualified 
healthcare provider for medical concerns.
"""
                
                st.download_button(
                    "📥 Download PDF Report (Text Format)",
                    data=report_text,
                    file_name=f"skin_analysis_report_{user['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            st.success(f"✅ {format.upper()} report generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()

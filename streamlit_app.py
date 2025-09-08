"""
Streamlit Web Application for Skin Allergy Risk Prediction
Modern, interactive interface for the XGBoost model
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

# Configure page
st.set_page_config(
    page_title="Skin Allergy Risk Prediction",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧴 Skin Allergy Risk Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model, preprocessor = load_model_and_preprocessor()
    
    if model is None or preprocessor is None:
        st.error("❌ Failed to load the trained model. Please ensure model files exist in the 'models' directory.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🔬 Allergy Risk Predictor")
    page = st.sidebar.selectbox("Choose a page:", ["🏠 Prediction", "📊 Model Info", "ℹ️ About"])
    
    if page == "🏠 Prediction":
        st.markdown("### Enter your information to get an allergy risk assessment")
        
        # Get input data
        input_data = create_input_form()
        
        # Prediction button
        if st.sidebar.button("🔮 Predict Risk Level", type="primary"):
            with st.spinner("Analyzing your data..."):
                risk_level, confidence, probabilities = make_prediction(model, preprocessor, input_data)
            
            if risk_level is not None:
                # Display results
                display_prediction_results(risk_level, confidence, probabilities)
                
                # Show recommendations
                st.subheader("💡 Personalized Recommendations")
                recommendations = generate_recommendations(risk_level, input_data)
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
                
                # Save prediction (optional)
                prediction_data = {
                    'timestamp': datetime.now().isoformat(),
                    'input_data': input_data,
                    'prediction': {
                        'risk_level': risk_level,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }
                }
                
                st.sidebar.download_button(
                    "📥 Download Results",
                    data=json.dumps(prediction_data, indent=2),
                    file_name=f"allergy_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Display sample input info
        st.markdown("---")
        st.markdown("### 📝 Quick Start Guide")
        st.markdown("""
        1. **Fill out the form** in the sidebar with your personal information
        2. **Adjust the sliders** to match your lifestyle and environmental exposure
        3. **Click 'Predict Risk Level'** to get your assessment
        4. **Review the recommendations** based on your risk level
        
        💡 **Tip**: The more accurate your input, the more reliable the prediction!
        """)
    
    elif page == "📊 Model Info":
        display_model_info(model)
    
    elif page == "ℹ️ About":
        display_about()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Skin Allergy Risk Prediction v1.0**")
    st.sidebar.markdown("Built with Streamlit & XGBoost")

if __name__ == "__main__":
    main()

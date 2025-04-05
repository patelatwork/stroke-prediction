import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# We'll use a simpler model approach without XGBoost
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.7rem;
        color: #0D47A1;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .risk-high {
        font-size: 2rem;
        color: #D32F2F;
        font-weight: bold;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .risk-medium {
        font-size: 2rem;
        color: #FF9800;
        font-weight: bold;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .risk-low {
        font-size: 2rem;
        color: #4CAF50;
        font-weight: bold;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0D47A1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .symptom-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .symptom-group-title {
        font-size: 1.3rem;
        color: #0D47A1;
        margin-bottom: 1rem;
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.5rem;
    }
    .checkbox-container label {
        font-weight: 500;
        color: #333;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .recommendation-card {
        background: #f1f8ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .footer {
        border-top: 1px solid #ddd;
        padding-top: 10px;
        font-size: 0.9rem;
        color: #666;
        text-align: center;
    }
    .tab-content {
        padding: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f4f5f7;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown("<h1 class='main-header'>Stroke Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Function to load and preprocess data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('dataset.csv')
    except:
        try:
            df = pd.read_csv('stroke_risk_dataset_v2.csv')
        except:
            # If neither file exists, create sample data
            st.warning("Dataset files not found. Using sample data instead.")
            # Create sample data
            np.random.seed(42)
            n_samples = 1000
            
            sample_data = {
                'age': np.random.uniform(20, 80, n_samples),
                'gender': np.random.choice(['Male', 'Female'], n_samples),
                'chest_pain': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'high_blood_pressure': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'irregular_heartbeat': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'shortness_of_breath': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'fatigue_weakness': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'dizziness': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'swelling_edema': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'neck_jaw_pain': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'excessive_sweating': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'persistent_cough': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'nausea_vomiting': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'chest_discomfort': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'cold_hands_feet': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'snoring_sleep_apnea': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'anxiety_doom': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            }
            
            # Calculate risk score (simplified approach for sample data)
            risk_factors = ['chest_pain', 'high_blood_pressure', 'irregular_heartbeat', 
                           'shortness_of_breath', 'dizziness']
            
            # Base risk on age and other factors
            risk_scores = sample_data['age'] / 100 * 30  # Age contributes up to 30%
            
            # Add contribution from other factors
            for factor in risk_factors:
                risk_scores += sample_data[factor] * 15  # Each factor can add up to 15%
            
            # Ensure risk is between 0 and 100
            risk_scores = np.clip(risk_scores, 0, 100)
            
            sample_data['stroke_risk_percentage'] = risk_scores
            sample_data['at_risk'] = (risk_scores >= 50).astype(int)
            
            df = pd.DataFrame(sample_data)
    
    # Clean gender values
    gender_corrections = {
        'Mlae': 'Male', 'Mael': 'Male', 'Mle': 'Male', 'M@le': 'Male', 'Mal': 'Male',
        'Femail': 'Female', 'Femle': 'Female', 'Fmale': 'Female', 'Fem@le': 'Female', 'Feamle': 'Female'
    }
    df['gender'] = df['gender'].replace(gender_corrections)
    
    # Drop rows with missing gender values
    df = df.dropna(subset=['gender'])
    
    # Clean numeric data
    def clean_numeric(value):
        if pd.isna(value): 
            return value
        if isinstance(value, str):
            import re
            numeric_str = re.findall(r'-?\d*\.?\d+', str(value))
            return float(numeric_str[0]) if numeric_str else np.nan
        return value
    
    # Identify and clean numeric columns
    numeric_columns = [col for col in df.columns if col != 'gender']
    
    for col in numeric_columns:
        df[col] = df[col].apply(clean_numeric)
    
    # Convert to numeric type
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure all numeric columns are float
    df[numeric_columns] = df[numeric_columns].astype('float64')
    
    # Handle missing values (if any)
    # First create a gender encoding for the imputer
    le = LabelEncoder()
    gender_encoded = le.fit_transform(df['gender'])
    df_with_encoded_gender = df[numeric_columns].copy()
    df_with_encoded_gender['gender_encoded'] = gender_encoded
    
    # Use the KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    numeric_cols_with_gender = list(df_with_encoded_gender.columns)
    imputed_data = imputer.fit_transform(df_with_encoded_gender)
    
    # Create imputed dataframe
    df_imputed = pd.DataFrame(imputed_data, columns=numeric_cols_with_gender, index=df.index)
    
    # Extract the gender_encoded column and convert back to original gender
    encoded_gender = df_imputed['gender_encoded'].values
    df_imputed = df_imputed.drop('gender_encoded', axis=1)
    
    # Add the original gender column back
    df_imputed['gender'] = le.inverse_transform(encoded_gender.astype(int))
    
    # Ensure at_risk is binary
    df_imputed['at_risk'] = (df_imputed['stroke_risk_percentage'] >= 50).astype(int)
    
    return df_imputed, le, imputer

# Function to train models
@st.cache_resource
def train_models(df):
    # Prepare features and target
    X = df.drop(['at_risk', 'stroke_risk_percentage', 'gender'], axis=1)
    X['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})
    y = df['at_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize base models
    base_models = {
        'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'lr': LogisticRegression(random_state=42)
    }
    
    # Train base models
    trained_models = {}
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Train meta-model (stacked model)
    meta_features = np.zeros((X_train.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(trained_models.items()):
        meta_features[:, i] = model.predict_proba(X_train)[:, 1]
    
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_features, y_train)
    
    return trained_models, meta_model, X_train.columns

# Function to make predictions using the stacked model
def predict_with_stacked_model(input_df, trained_models, meta_model, columns):
    # Ensure input_df has the same columns as training data
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[columns]
    
    # Get predictions from base models
    meta_features = np.zeros((1, len(trained_models)))
    
    for i, (name, model) in enumerate(trained_models.items()):
        meta_features[0, i] = model.predict_proba(input_df)[0, 1]
    
    # Get prediction from meta-model
    risk_prob = meta_model.predict_proba(meta_features)[0, 1]
    risk_percentage = risk_prob * 100
    risk_class = 1 if risk_percentage >= 50 else 0
    
    return risk_percentage, risk_class, meta_features[0]

# Load data and train models
with st.spinner("Loading data and training models... This might take a minute."):
    df, label_encoder, imputer = load_data()
    trained_models, meta_model, columns = train_models(df)

# Create tabs for the application
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "About"])

with tab1:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Patient Information</h3>", unsafe_allow_html=True)
    
    # Create three columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic information
        st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
        st.markdown("<h4 class='symptom-group-title'>Basic Information</h4>", unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Empty space for layout balance
        st.markdown("<div class='symptom-card' style='visibility: hidden;'>", unsafe_allow_html=True)
        st.markdown("<h4 class='symptom-group-title'>Hidden</h4>", unsafe_allow_html=True)
        st.write(" ")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Symptom selection using checkboxes
    st.markdown("<h3 class='sub-header'>Symptoms</h3>", unsafe_allow_html=True)
    
    # Create three columns for symptoms layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Vital symptoms
        st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
        st.markdown("<h4 class='symptom-group-title'>Vital Symptoms</h4>", unsafe_allow_html=True)
        
        high_blood_pressure = 1.0 if st.checkbox("High Blood Pressure") else 0.0
        irregular_heartbeat = 1.0 if st.checkbox("Irregular Heartbeat") else 0.0
        chest_pain = 1.0 if st.checkbox("Chest Pain") else 0.0
        shortness_of_breath = 1.0 if st.checkbox("Shortness of Breath") else 0.0
        chest_discomfort = 1.0 if st.checkbox("Chest Discomfort") else 0.0
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Secondary symptoms
        st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
        st.markdown("<h4 class='symptom-group-title'>Secondary Symptoms</h4>", unsafe_allow_html=True)
        
        fatigue_weakness = 1.0 if st.checkbox("Fatigue/Weakness") else 0.0
        dizziness = 1.0 if st.checkbox("Dizziness") else 0.0
        swelling_edema = 1.0 if st.checkbox("Swelling/Edema") else 0.0
        neck_jaw_pain = 1.0 if st.checkbox("Neck/Jaw Pain") else 0.0
        excessive_sweating = 1.0 if st.checkbox("Excessive Sweating") else 0.0
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        # Other symptoms
        st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
        st.markdown("<h4 class='symptom-group-title'>Other Symptoms</h4>", unsafe_allow_html=True)
        
        persistent_cough = 1.0 if st.checkbox("Persistent Cough") else 0.0
        nausea_vomiting = 1.0 if st.checkbox("Nausea/Vomiting") else 0.0
        cold_hands_feet = 1.0 if st.checkbox("Cold Hands/Feet") else 0.0
        snoring_sleep_apnea = 1.0 if st.checkbox("Snoring/Sleep Apnea") else 0.0
        anxiety_doom = 1.0 if st.checkbox("Anxiety/Feeling of Doom") else 0.0
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Predict button centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Predict Stroke Risk")
    
    st.markdown("---")
    
    if predict_button:
        # Create input data for prediction
        input_data = {
            'age': age,
            'chest_pain': chest_pain,
            'high_blood_pressure': high_blood_pressure,
            'irregular_heartbeat': irregular_heartbeat,
            'shortness_of_breath': shortness_of_breath,
            'fatigue_weakness': fatigue_weakness,
            'dizziness': dizziness,
            'swelling_edema': swelling_edema,
            'neck_jaw_pain': neck_jaw_pain,
            'excessive_sweating': excessive_sweating,
            'persistent_cough': persistent_cough,
            'nausea_vomiting': nausea_vomiting,
            'chest_discomfort': chest_discomfort,
            'cold_hands_feet': cold_hands_feet,
            'snoring_sleep_apnea': snoring_sleep_apnea,
            'anxiety_doom': anxiety_doom,
            'gender_encoded': 1 if gender == 'Female' else 0
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        risk_percentage, risk_class, model_contributions = predict_with_stacked_model(
            input_df, trained_models, meta_model, columns
        )
        
        # Display results
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<h3 class='sub-header'>Prediction Results</h3>", unsafe_allow_html=True)
            
            # Create a gauge chart for risk visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Stroke Risk Percentage", 'font': {'size': 24, 'color': '#0D47A1'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#4CAF50", 'thickness': 0.75},
                        {'range': [30, 70], 'color': "#FF9800", 'thickness': 0.75},
                        {'range': [70, 100], 'color': "#D32F2F", 'thickness': 0.75}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=350,
                margin=dict(l=30, r=30, t=50, b=30),
                paper_bgcolor="white",
                font={'color': "#555", 'family': "Arial"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display risk category
            if risk_percentage >= 70:
                st.markdown(f"<p class='risk-high'>High Risk: {risk_percentage:.1f}%</p>", unsafe_allow_html=True)
                st.warning("⚠ The patient is at HIGH RISK of stroke. Please consult a healthcare provider immediately.")
            elif risk_percentage >= 30:
                st.markdown(f"<p class='risk-medium'>Medium Risk: {risk_percentage:.1f}%</p>", unsafe_allow_html=True)
                st.info("ℹ The patient is at MEDIUM RISK of stroke. Regular monitoring and lifestyle changes are recommended.")
            else:
                st.markdown(f"<p class='risk-low'>Low Risk: {risk_percentage:.1f}%</p>", unsafe_allow_html=True)
                st.success("✅ The patient is at LOW RISK of stroke.")
        
        with col2:
            st.markdown("<h3 class='sub-header'>Risk Factors</h3>", unsafe_allow_html=True)
            
            # Display major risk factors
            risk_factors = []
            if high_blood_pressure >= 0.5: risk_factors.append(("High Blood Pressure", high_blood_pressure))
            if irregular_heartbeat >= 0.5: risk_factors.append(("Irregular Heartbeat", irregular_heartbeat))
            if chest_pain >= 0.5: risk_factors.append(("Chest Pain", chest_pain))
            if shortness_of_breath >= 0.5: risk_factors.append(("Shortness of Breath", shortness_of_breath))
            if dizziness >= 0.5: risk_factors.append(("Dizziness", dizziness))
            if swelling_edema >= 0.5: risk_factors.append(("Swelling/Edema", swelling_edema))
            if age >= 65: risk_factors.append(("Advanced Age", 1.0))
            
            if risk_factors:
                st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                st.write("### Major risk factors identified:")
                for factor, value in sorted(risk_factors, key=lambda x: x[1], reverse=True):
                    st.write(f"• *{factor}*")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='recommendation-card' style='background-color: #e8f5e9;'>", unsafe_allow_html=True)
                st.write("### No major risk factors identified")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display model contributions
            st.markdown("<h4>Model Contributions</h4>", unsafe_allow_html=True)
            model_names = list(trained_models.keys())
            
            # Create horizontal bar chart of model contributions
            fig = px.bar(
                x=model_contributions,
                y=model_names,
                orientation='h',
                labels={'x': 'Contribution to Risk Score', 'y': 'Model'},
                title='Model Contributions to Risk Assessment',
                color=model_contributions,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                title_font=dict(size=18, color='#0D47A1'),
                font=dict(size=14),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Recommendations section
        st.markdown("<h3 class='sub-header'>Recommendations</h3>", unsafe_allow_html=True)
        
        general_recommendations = [
            "Maintain a healthy diet rich in fruits, vegetables, and whole grains",
            "Exercise regularly (at least 150 minutes of moderate activity per week)",
            "Manage stress through relaxation techniques and adequate sleep",
            "Avoid smoking and limit alcohol consumption"
        ]
        
        risk_specific_recommendations = []
        if high_blood_pressure >= 0.5:
            risk_specific_recommendations.append("Monitor blood pressure regularly and follow medical advice for management")
        if irregular_heartbeat >= 0.5:
            risk_specific_recommendations.append("Consult a cardiologist for evaluation and management of arrhythmia")
        if chest_pain >= 0.5 or chest_discomfort >= 0.5:
            risk_specific_recommendations.append("Seek immediate medical attention for chest pain or discomfort")
        if age >= 65:
            risk_specific_recommendations.append("Schedule regular check-ups with healthcare providers")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
            st.subheader("General Health Recommendations")
            for rec in general_recommendations:
                st.write(f"• {rec}")
            st.markdown("</div>", unsafe_allow_html=True)
                
        with col2:
            st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
            st.subheader("Risk-Specific Recommendations")
            if risk_specific_recommendations:
                for rec in risk_specific_recommendations:
                    st.write(f"• {rec}")
            else:
                st.write("No specific recommendations based on risk factors.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Data Insights</h3>", unsafe_allow_html=True)
    
    # Key statistics with improved styling
    st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        total_cases = len(df)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{total_cases:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Cases</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        high_risk_cases = df['at_risk'].sum()
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{high_risk_cases:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>High Risk Cases</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        risk_percentage = (high_risk_cases / total_cases) * 100
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{risk_percentage:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Percentage at High Risk</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Keep the rest of tab2 content mostly unchanged
    # Distribution by gender
    st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
    st.subheader("Risk Distribution by Gender")
    try:
        gender_risk = df.groupby(['gender', 'at_risk']).size().unstack()
        gender_risk.columns = ['Low Risk', 'High Risk']
        
        fig = px.bar(
            gender_risk, 
            barmode='group',
            title="Risk Distribution by Gender",
            color_discrete_sequence=['#4CAF50', '#D32F2F']
        )
        fig.update_layout(
            height=400,
            title_font=dict(size=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Unable to create gender distribution chart. There might be an issue with the data structure.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Age vs Risk
    st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
    st.subheader("Age vs Stroke Risk")
    
    try:
        fig = px.scatter(
            df, x='age', y='stroke_risk_percentage', 
            color='at_risk', 
            color_discrete_sequence=['#4CAF50', '#D32F2F'],
            title="Age vs Stroke Risk",
            labels={'age': 'Age', 'stroke_risk_percentage': 'Risk Percentage', 'at_risk': 'High Risk'}
        )
        fig.update_layout(
            height=400,
            title_font=dict(size=20),
            plot_bgcolor='rgba(0,0,0,0.02)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Unable to create age vs risk chart. There might be an issue with the data structure.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Correlation heatmap
    st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
    st.subheader("Symptom Correlation Analysis")
    
    try:
        # Correlation matrix
        corr_matrix = df.drop(['gender'], axis=1).corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        fig.update_layout(
            height=600,
            title_font=dict(size=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Unable to create correlation heatmap. There might be an issue with the data structure.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>About the Stroke Risk Prediction System</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='symptom-card'>", unsafe_allow_html=True)
    st.markdown("""
    This application uses machine learning to predict the risk of stroke based on patient information and symptoms. 
    The system employs a stacked ensemble model that combines multiple ML algorithms to provide accurate predictions.
    
    ### Features Used for Prediction
    
    - *Patient Information*: Age and gender
    - *Vital Symptoms*: High blood pressure, irregular heartbeat, chest pain, shortness of breath
    - *Secondary Symptoms*: Fatigue, dizziness, swelling, pain in neck/jaw, sweating
    - *Other Symptoms*: Cough, nausea, cold extremities, sleep apnea, anxiety
    
    ### Model Information
    
    The system utilizes a stacked ensemble approach that combines three powerful models:
    
    1. *Gradient Boosting Classifier*: Effective for handling complex relationships
    2. *Random Forest*: Robust against overfitting and handles non-linear relationships
    3. *Logistic Regression*: Interpretable model that serves as a baseline
    
    The predictions from these models are combined using a meta-model to produce the final risk assessment.
    
    ### Interpretation of Results
    
    - *Low Risk (<30%)*: Low probability of stroke
    - *Medium Risk (30-70%)*: Moderate probability of stroke
    - *High Risk (>70%)*: High probability of stroke
    
    ### Disclaimer
    
    This tool is for informational purposes only and does not constitute medical advice. 
    Always consult with qualified healthcare providers for diagnosis and treatment decisions.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div class='footer'>© 2025 Stroke Risk Prediction System | All Rights Reserved |Created By Dhruv Patel && Aadi Patel</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("<p class='info-text'>This application is intended for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment.</p>", unsafe_allow_html=True)
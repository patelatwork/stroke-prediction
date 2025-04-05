import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("Stroke Risk Prediction App")
st.write("Enter patient information to predict stroke risk")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    
    # Basic information
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    st.subheader("Symptoms")
    
    # Create sliders for all symptoms
    chest_pain = st.slider("Chest Pain", 0.0, 1.0, 0.0)
    high_blood_pressure = st.slider("High Blood Pressure", 0.0, 1.0, 0.0)
    irregular_heartbeat = st.slider("Irregular Heartbeat", 0.0, 1.0, 0.0)
    shortness_of_breath = st.slider("Shortness of Breath", 0.0, 1.0, 0.0)
    fatigue_weakness = st.slider("Fatigue/Weakness", 0.0, 1.0, 0.0)
    dizziness = st.slider("Dizziness", 0.0, 1.0, 0.0)
    swelling_edema = st.slider("Swelling/Edema", 0.0, 1.0, 0.0)
    neck_jaw_pain = st.slider("Neck/Jaw Pain", 0.0, 1.0, 0.0)
    excessive_sweating = st.slider("Excessive Sweating", 0.0, 1.0, 0.0)
    persistent_cough = st.slider("Persistent Cough", 0.0, 1.0, 0.0)
    nausea_vomiting = st.slider("Nausea/Vomiting", 0.0, 1.0, 0.0)
    chest_discomfort = st.slider("Chest Discomfort", 0.0, 1.0, 0.0)
    cold_hands_feet = st.slider("Cold Hands/Feet", 0.0, 1.0, 0.0)
    snoring_sleep_apnea = st.slider("Snoring/Sleep Apnea", 0.0, 1.0, 0.0)
    anxiety_doom = st.slider("Anxiety/Feeling of Doom", 0.0, 1.0, 0.0)

if st.button("Predict Stroke Risk"):
    # Create a DataFrame with the input values
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
        'anxiety_doom': anxiety_doom
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Add gender encoding
    le = LabelEncoder()
    le.fit(['Male', 'Female'])
    input_df['gender_encoded'] = le.transform([gender])

    # Create and train a new model with the same parameters
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # Load your training data
    df = pd.read_csv('dataset.csv')
    
    # Prepare features and target
    X = df.drop(['at_risk', 'stroke_risk_percentage', 'gender'], axis=1)
    y = (df['stroke_risk_percentage'] >= 50).astype(int)
    
    # Add gender encoding to training data
    X['gender_encoded'] = le.fit_transform(df['gender'])
    
    # Train the model
    gb_model.fit(X, y)

    # Make prediction
    prediction = gb_model.predict_proba(input_df)[0]
    risk_percentage = prediction[1] * 100

    # Display results
    with col2:
        st.subheader("Prediction Results")
        
        # Create a progress bar for risk visualization
        st.progress(int(risk_percentage))
        
        if risk_percentage >= 50:
            st.error(f"High Risk: {risk_percentage:.1f}%")
            st.write("⚠️ The patient is at high risk of stroke. Please consult a healthcare provider immediately.")
        else:
            st.success(f"Low Risk: {risk_percentage:.1f}%")
            st.write("✅ The patient is at low risk of stroke.")
        
        # Display risk factors
        st.subheader("Key Risk Factors")
        risk_factors = []
        if high_blood_pressure > 0.5: risk_factors.append("High Blood Pressure")
        if irregular_heartbeat > 0.5: risk_factors.append("Irregular Heartbeat")
        if chest_pain > 0.5: risk_factors.append("Chest Pain")
        
        if risk_factors:
            st.write("Major risk factors present:")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("No major risk factors identified.")

# Add footer
st.markdown("---")
st.markdown("### Disclaimer")
st.write("This tool is for informational purposes only and should not be used as a substitute for professional medical advice.")
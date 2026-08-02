import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase a tourism package based on their details.
Enter the customer data below to get a prediction.
""")

age = st.number_input("Age", 18, 65, 30, 1)
type_of_contact = st.selectbox("Contact Type", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of People Visiting", 1, 10, 2, 1)
preferred_property_star = st.number_input("Preferred Property Star", 1, 5, 3, 1)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips", 1, 10, 2, 1)
passport = st.selectbox("Passport", ["No", "Yes"])
own_car = st.selectbox("Own Car", ["No", "Yes"])
number_of_children_visiting = st.number_input("Number of Children (below age 5) Visiting", 0, 10, 0, 1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 0, 100000, 50000, 1000)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", 1, 5, 3, 1)
product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Standard", "Basic", "Super Deluxe", "King"])
number_of_followups = st.number_input("Number of Follow-ups", 0, 10, 0, 1)
duration_of_pitch = st.number_input("Duration of Pitch", 1, 150, 15, 1)

input_data = pd.DataFrame([{
    # "CustomerID": customer_id, # Removed as it's dropped during preprocessing
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": 1 if city_tier == "Tier 1" else (2 if city_tier == "Tier 2" else 3),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0, # Convert 'Yes'/'No' to 1/0
    "OwnCar": 1 if own_car == "Yes" else 0,     # Convert 'Yes'/'No' to 1/0
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch
}])

# Convert categorical features to match the exact format used during training
# These conversions are for 'Passport' and 'OwnCar' which are binary but taken as string inputs

if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

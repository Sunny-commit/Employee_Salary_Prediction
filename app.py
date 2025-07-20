import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
# These inputs should match the features used for training the model.
# Based on the training data, the features are:
# 'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
# 'occupation', 'relationship', 'race', 'gender', 'capital-gain',
# 'capital-loss', 'hours-per-week', 'native-country'

# Example of how to add inputs based on the actual features:
age = st.sidebar.slider("Age", 17, 75, 30) # Adjusted range based on outlier removal
# Note: workclass, marital-status, occupation, relationship, race, gender, native-country
# are encoded using LabelEncoder. For a user-friendly app, you would need
# to reverse the encoding or use OneHotEncoding and provide dropdowns for categories.
# For simplicity in this example, we'll add a few representative features.
# In a real application, you would need to handle all features used in training.

educational_num = st.sidebar.slider("Educational Number", 5, 16, 9) # Adjusted range based on outlier removal
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40) # Assuming a reasonable range

# Note: Other features like 'fnlwgt', 'capital-gain', 'capital-loss',
# 'workclass', 'marital-status', 'occupation', 'relationship', 'race',
# 'gender', and 'native-country' would also need to be included in the
# input_df and potentially handled with appropriate input widgets and
# preprocessing (like the LabelEncoding used in the notebook)

# Build input DataFrame (⚠️ must match preprocessing of your training data)
# This input_df needs to match the columns and their data types after preprocessing
# done in the training notebook. This is a simplified example.
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [educational_num],
    'hours-per-week': [hours_per_week]
    # Add other features here with appropriate dummy values or user inputs
    # based on the encoding/preprocessing steps in the notebook
})

st.write("### 🔎 Input Data (Simplified Example)")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class (Simplified Model)"):
    # Note: This prediction will likely be inaccurate as the input_df
    # does not contain all the features used in training the model.
    # You would need to replicate the full preprocessing pipeline here.
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}. Ensure input features match the trained model.")


# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Note: The batch_data from the uploaded file also needs to undergo
    # the same preprocessing steps (handling '?', outlier removal, LabelEncoding)
    # as the training data before making predictions.
    # This part is missing in the current app.py

    try:
        # This prediction will likely fail if the uploaded batch_data
        # does not have the same columns and format as the training data
        # after preprocessing.
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Batch prediction error: {e}. Ensure uploaded data matches the expected format and features after preprocessing.")

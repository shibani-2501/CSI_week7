import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.random.rand(100, 3) * 10 # 100 samples, 3 features
y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 2 + 5

df = pd.DataFrame(X, columns=['Feature_A', 'Feature_B', 'Feature_C'])
df['Target'] = y

model = LinearRegression()
model.fit(X, y)

model_filename = 'linear_regression_model.pkl'
joblib.dump(model, model_filename)

st.set_page_config(
    page_title="ML Model Deployment App",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📊 Machine Learning Model Deployment")
st.markdown("""
This application demonstrates how to deploy a machine learning model using Streamlit.
You can input values for different features and get a prediction from a simulated model.
""")
try:
    loaded_model = joblib.load(model_filename)
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the same directory.")
    st.stop() # Stop the app if the model isn't found

st.sidebar.header("Input Features")

feature_a = st.sidebar.slider("Feature A", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
feature_b = st.sidebar.slider("Feature B", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
feature_c = st.sidebar.slider("Feature C", min_value=0.0, max_value=20.0, value=3.0, step=0.1)

input_data = pd.DataFrame([[feature_a, feature_b, feature_c]],
                          columns=['Feature_A', 'Feature_B', 'Feature_C'])

st.subheader("Your Input Data")
st.write(input_data)

if st.button("Get Prediction"):
    prediction = loaded_model.predict(input_data)
    st.subheader("Prediction")
    st.success(f"The predicted value is: **{prediction[0]:.2f}**")

    st.markdown("---")
    st.subheader("Model Insights (Simulated)")
    st.markdown("""
    Below are some simulated visualizations to help understand the model's behavior.
    In a real application, these would represent actual model diagnostics,
    such as feature importance, partial dependence plots, or error distributions.
    """)

    st.write("#### Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': ['Feature A', 'Feature B', 'Feature C'],
        'Importance': np.abs(loaded_model.coef_) # Using absolute coefficients as proxy
    }).sort_values(by='Importance', ascending=False)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax1, palette='viridis')
    ax1.set_title("Simulated Feature Importance")
    ax1.set_xlabel("Importance (Absolute Coefficient Value)")
    ax1.set_ylabel("Feature")
    st.pyplot(fig1)
    st.markdown("This plot shows which features have a stronger influence on the prediction based on the model's coefficients.")

    st.write("#### Prediction Distribution")
    # For this dummy model, let's show a histogram of past predictions or target values
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Target'], kde=True, ax=ax2, color='skyblue')
    ax2.axvline(prediction[0], color='red', linestyle='--', label=f'Your Prediction ({prediction[0]:.2f})')
    ax2.set_title("Distribution of Target Values (Historical Data)")
    ax2.set_xlabel("Target Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    st.pyplot(fig2)
    st.markdown("This histogram shows the distribution of target values from the training data, with your current prediction highlighted.")

    
    st.write("#### Model Coefficients")
    coefficients_df = pd.DataFrame({
        'Feature': ['Feature A', 'Feature B', 'Feature C'],
        'Coefficient': loaded_model.coef_
    })
    st.table(coefficients_df)
    st.markdown("The coefficients indicate how much the target variable is expected to change for a one-unit increase in each feature, holding other features constant.")

st.markdown("---")
st.markdown("### How to Use This App")
st.markdown("""
1.  **Adjust the sliders** in the sidebar to change the values of 'Feature A', 'Feature B', and 'Feature C'.
2.  Click the **"Get Prediction"** button to see the model's output based on your input.
3.  Explore the **"Model Insights"** section for visualizations that help explain the prediction.
""")

st.markdown("---")


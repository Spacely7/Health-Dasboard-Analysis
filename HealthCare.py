import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare_dataset.csv")
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Stay Duration (days)'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df['Billing Category'] = pd.cut(df['Billing Amount'],
                                    bins=[-float('inf'), 10000, 30000, float('inf')],
                                    labels=['Low', 'Medium', 'High'])
    return df

df = load_data()

st.title("🏥 Healthcare Insights Dashboard")

# KPI metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", df.shape[0])
col2.metric("Avg. Billing Amount", f"${df['Billing Amount'].mean():,.2f}")
col3.metric("Avg. Stay Duration", f"{df['Stay Duration (days)'].mean():.1f} days")

# Alert for data issues
neg_billing_count = (df['Billing Amount'] < 0).sum()
if neg_billing_count > 0:
    st.warning(f"⚠️ {neg_billing_count} records have **negative billing amounts**. Please review them.")

# Sidebar filters
st.sidebar.header("📌 Filters")
condition = st.sidebar.multiselect("Filter by Medical Condition", df['Medical Condition'].unique())
gender = st.sidebar.multiselect("Filter by Gender", df['Gender'].unique())

filtered_df = df.copy()
if condition:
    filtered_df = filtered_df[filtered_df['Medical Condition'].isin(condition)]
if gender:
    filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]

# Visualizations
st.header("🔹 Patient Demographics & Distribution")

# Gender Pie Chart
fig_gender = px.pie(filtered_df, names='Gender', title='Gender Distribution', hole=0.4)
st.plotly_chart(fig_gender, use_container_width=True)

# Medical Condition Bar Chart
st.subheader("Top Medical Conditions")
st.bar_chart(filtered_df['Medical Condition'].value_counts().head(10))

# Insurance Provider
st.subheader("Insurance Providers")
st.bar_chart(filtered_df['Insurance Provider'].value_counts())

# Billing Pie Chart
st.subheader("Billing Categories")
fig_bill = px.pie(filtered_df, names='Billing Category', title='Billing Amount Categories', hole=0.4)
st.plotly_chart(fig_bill, use_container_width=True)

# Test Results
st.subheader("Test Results Overview")
st.bar_chart(filtered_df['Test Results'].value_counts())

# Admission Types
st.subheader("Admission Types")
st.bar_chart(filtered_df['Admission Type'].value_counts())

# Medication Distribution
st.subheader("Most Prescribed Medications")
st.bar_chart(filtered_df['Medication'].value_counts().head(10))

# Average Stay by Condition
st.subheader("Avg. Hospital Stay per Condition")
avg_stay = filtered_df.groupby('Medical Condition')['Stay Duration (days)'].mean().sort_values(ascending=False)
st.bar_chart(avg_stay)

# 📅 Time Series: Admissions per Month
st.header("📆 Admission Trends Over Time")
admission_trend = filtered_df.groupby(filtered_df['Date of Admission'].dt.to_period('M')).size()
admission_trend.index = admission_trend.index.astype(str)
fig_admit = px.line(x=admission_trend.index, y=admission_trend.values,
                    labels={'x': 'Month', 'y': 'Admissions'}, title="Monthly Admissions Over Time")
st.plotly_chart(fig_admit, use_container_width=True)

# 💵 Billing Trends Over Time
st.subheader("Average Billing Over Time")
billing_trend = filtered_df.groupby(filtered_df['Date of Admission'].dt.to_period('M'))['Billing Amount'].mean()
billing_trend.index = billing_trend.index.astype(str)
fig_billing = px.line(x=billing_trend.index, y=billing_trend.values,
                      labels={'x': 'Month', 'y': 'Avg Billing'}, title="Avg. Billing Amount Over Time")
st.plotly_chart(fig_billing, use_container_width=True)

# 📊 Diagnostic Summary
st.header("🧠 Test Results Summary by Condition")
summary = filtered_df.groupby(['Medical Condition', 'Test Results']).size().unstack(fill_value=0)
st.dataframe(summary)

# 📥 Downloadable Filtered Report
st.header("📥 Download Filtered Data")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered_df)

st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='filtered_healthcare_data.csv',
    mime='text/csv',
)

# Show preview
st.header("🔍 Preview Data")
st.dataframe(filtered_df.head(100))


st.header("🔮 Predict Test Result for New Patient")

# Select features & encode
ml_df = df[['Age', 'Gender', 'Medical Condition', 'Insurance Provider',
            'Billing Amount', 'Admission Type', 'Medication', 'Test Results']].copy()

# Encode categoricals
label_encoders = {}
for col in ml_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    ml_df[col] = le.fit_transform(ml_df[col])
    label_encoders[col] = le

# Train model
X = ml_df.drop('Test Results', axis=1)
y = ml_df['Test Results']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# User input form
with st.form("prediction_form"):
    st.subheader("Enter New Patient Details")
    age = st.slider("Age", 13, 90, 45)
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    condition = st.selectbox("Medical Condition", label_encoders['Medical Condition'].classes_)
    insurance = st.selectbox("Insurance Provider", label_encoders['Insurance Provider'].classes_)
    billing = st.number_input("Billing Amount", value=25000.0)
    admission_type = st.selectbox("Admission Type", label_encoders['Admission Type'].classes_)
    medication = st.selectbox("Medication", label_encoders['Medication'].classes_)

    submitted = st.form_submit_button("Predict Test Result")

    if submitted:
        # Encode inputs
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [label_encoders['Gender'].transform([gender])[0]],
            'Medical Condition': [label_encoders['Medical Condition'].transform([condition])[0]],
            'Insurance Provider': [label_encoders['Insurance Provider'].transform([insurance])[0]],
            'Billing Amount': [billing],
            'Admission Type': [label_encoders['Admission Type'].transform([admission_type])[0]],
            'Medication': [label_encoders['Medication'].transform([medication])[0]]
        })

        # Predict
        prediction = rf.predict(input_data)
        predicted_label = label_encoders['Test Results'].inverse_transform(prediction)[0]

        st.success(f"🧪 Predicted Test Result: **{predicted_label}**")

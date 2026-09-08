import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Resume Screening & Classification System")
st.markdown("Automate candidate evaluation using Machine Learning.")

# Sidebar - Dataset Upload
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload 'resumedataset.csv'", type=["csv"])

@st.cache_resource
def train_model(dataframe):
    df = dataframe.copy()
    
    # Preprocessing
    df['Certifications'] = df['Certifications'].fillna('None')
    df['Target'] = df['Recruiter Decision'].apply(lambda x: 1 if str(x).strip().lower() == 'hire' else 0)
    
    feature_cols = [
        'Skills', 'Experience (Years)', 'Education', 
        'Certifications', 'Job Role', 'Salary Expectation ($)', 
        'Projects Count', 'AI Score (0-100)'
    ]
    
    X = df[feature_cols]
    y = df['Target']
    
    categorical_cols = ['Skills', 'Education', 'Certifications', 'Job Role']
    numerical_cols = ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count', 'AI Score (0-100)']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    
    # Calculate Metrics
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return pipeline, accuracy, cm, df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model, accuracy, cm, cleaned_df = train_model(df)
    
    st.sidebar.success(f"Model Trained! Accuracy: **{accuracy * 100:.1f}%**")
    
    # Tabs Layout
    tab1, tab2, tab3 = st.tabs(["🎯 Screen Candidate", "📊 Dataset Analytics", "📈 Model Performance"])
    
    # ----------------------------------------------------
    # TAB 1: Real-time Candidate Screening Form
    # ----------------------------------------------------
    with tab1:
        st.subheader("Candidate Information Form")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_role = st.selectbox("Target Job Role", df['Job Role'].unique())
            education = st.selectbox("Highest Education", df['Education'].unique())
            skills = st.text_input("Skills (comma-separated)", "Python, TensorFlow, SQL, Machine Learning")
            certifications = st.selectbox("Certifications", ["None"] + list(df['Certifications'].dropna().unique()))
            
        with col2:
            experience = st.slider("Experience (Years)", 0, 20, 3)
            projects_count = st.slider("Projects Count", 0, 15, 4)
            ai_score = st.slider("Initial AI Match Score (0-100)", 0, 100, 75)
            salary_exp = st.number_input("Salary Expectation ($)", min_value=20000, max_value=250000, value=85000, step=5000)

        st.markdown("---")
        
        if st.button("Screen Candidate", type="primary"):
            input_data = pd.DataFrame([{
                'Skills': skills,
                'Experience (Years)': experience,
                'Education': education,
                'Certifications': certifications,
                'Job Role': job_role,
                'Salary Expectation ($)': salary_exp,
                'Projects Count': projects_count,
                'AI Score (0-100)': ai_score
            }])
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 1:
                    st.success("### Recommendation: HIRE")
                else:
                    st.error("### Recommendation: REJECT")
                    
            with res_col2:
                st.metric(label="Hiring Confidence Score", value=f"{probability * 100:.1f}%")

    # ----------------------------------------------------
    # TAB 2: Exploratory Data Analysis
    # ----------------------------------------------------
    with tab2:
        st.subheader("Dataset Preview & Distribution")
        st.dataframe(cleaned_df.head(10), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Recruiter Decisions Count**")
            st.bar_chart(cleaned_df['Recruiter Decision'].value_counts())
            
        with c2:
            st.markdown("**Experience vs AI Score**")
            fig, ax = plt.subplots()
            sns.scatterplot(data=cleaned_df, x='Experience (Years)', y='AI Score (0-100)', hue='Recruiter Decision', ax=ax)
            st.pyplot(fig)

    # ----------------------------------------------------
    # TAB 3: Model Performance Metrics
    # ----------------------------------------------------
    with tab3:
        st.subheader("Random Forest Classifier Evaluation")
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Test Accuracy Score", f"{accuracy * 100:.2f}%")
            
        with m2:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Reject', 'Hire'], yticklabels=['Reject', 'Hire'], ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

else:
    st.info("👈 Please upload the `resumedataset.csv` file from the sidebar to initialize the screening model.")
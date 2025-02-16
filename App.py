import streamlit as st
import joblib
import pdfminer.high_level
import sqlite3
import pandas as pd
import base64
import os
import time
import random
import datetime
import os
os.environ["PAFY_BACKEND"] = "internal"  # Ensures it works without youtube-dl
import pafy  
import plotly.express as px  # For visualizations
from streamlit_tags import st_tags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos


# Load the model
classifier = joblib.load('tfidf_classifier.pkl')

# Connect to the database
connection = sqlite3.connect("resume_data.db", check_same_thread=False)
cursor = connection.cursor()

# Create a table to store the resume data
cursor.execute(
    """CREATE TABLE IF NOT EXISTS user_data (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       name TEXT,
       email TEXT,
       resume_score TEXT,
       timestamp TEXT,
       pages TEXT,
       predicted_field TEXT,
       user_level TEXT,
       skills TEXT,
       recommended_skills TEXT,
       recommended_courses TEXT
    )"""
)
connection.commit()

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = pdfminer.high_level.extract_text(file_path)
    return text

# Load category mapping (job_position_name <-> category codes)
category_mapping = pd.read_csv("category_mapping.csv", index_col=0).iloc[:,0].to_dict()

def predict_category(text):
    """Predict the job category and convert it back to the job position name"""
    prediction_index = classifier.predict([text])[0]  # Get numeric prediction
    predicted_category = category_mapping.get(prediction_index, "Unknown")  # Convert number back to string
    return predicted_category

# Function to recommend courses based on the predicted category
def get_courses(predicted_field):
    """Returns recommended courses based on the job prediction"""
    course_dict = {
        "Data Science": ds_course,
        "Web Development": web_course,
        "Android Development": android_course,
        "IOS Development": ios_course,
        "UI/UX Development": uiux_course
    }
    return course_dict.get(predicted_field, [])

# Function to analyze the resume and recommend courses
def analyze_resume(text):
    # Extract the text from the resume
    text = extract_text_from_pdf(text)
    predicted_category = predict_category(text)
    recommended_courses = get_courses(predicted_category)

    return {
        "resume_text": text,
        "predicted_field": predicted_category,
        "recommended_courses": recommended_courses
    }

# Function to save the user data to the database
def insert_data(name, email, resume_score, timestamp, pages, predicted_field, user_level, sills, recommended_skills, recommended_courses):
    cursor.execute(
        """INSERT INTO user_data (name, email, resume_score, timestamp, pages, predicted_field, user_level, skills, recommended_skills, recommended_courses)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, email, resume_score, timestamp, pages, predicted_field, user_level, sills, recommended_skills, recommended_courses)
    )
    connection.commit()

# Function to Calculate Resume Score
def calculate_resume_score(resume_text, predicted_field):
    score = 0
    # Convert resume to lowercase
    resume_lower = resume_text.lower()
    # Score based on skills matching he predicted field
    field_keywords = {
    "Site Engineer": "civil engineering autocad project management construction structural design",
    "Civil Engineer": "structural analysis autocad civil 3d surveying reinforced concrete",
    "HR Officer": "recruitment performance management training payroll employee relations",
    "Project Coordinator (Civil)": "construction management budgeting autocad project planning ms project",
    "Business Development Executive": "sales market research client relations negotiation lead generation",
    "Marketing Officer": "digital marketing seo social media brand management market analysis",
    "AI Engineer": "machine learning deep learning python tensorflow nlp computer vision artificial intelligence",
    "Data Engineer": "sql etl big data spark hadoop data pipelines cloud databases",
    "Data Science Engineer": "data analysis pandas numpy scikit-learn deep learning ml algorithms data visualization",
    "Senior iOS Engineer": "swift xcode ios development ui ux core data apple guidelines"
    }

    if predicted_field not in field_keywords:
        return 50 # Default score
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_lower, field_keywords[predicted_field]])
    score = cosine_similarity(tfidf_matrix)[0,1]
    # Ensure score is within 0-100
    score = min(int(score), 100)
    return score

# Streamlit UI
st.set_page_config(page_title='AI Resume Analyzer', page_icon= '📄', layout='wide', initial_sidebar_state='expanded')

st.sidebar.markdown("## User Dashboard")
activities = ['User', 'Admin']
choice = st.sidebar.selectbox("Choose Role:",  activities)


if choice == "User":
    uploaded_file = st.file_uploader("📂 Upload Resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
        temp_path = "temp_resume.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        resume_text = extract_text_from_pdf(temp_path)
        predicted_field = predict_category(resume_text)
        recommended_courses = get_courses(predicted_field)
        resume_score = calculate_resume_score(resume_text, predicted_field)
        os.remove(temp_path)  # Clean up

        st.subheader("🔍 Predicted Job Field:")
        st.success(predicted_field)

        st.subheader("📊 Resume Score:")
        st.progress(resume_score / 100)
        st.write(f"Your resume score is **{resume_score}/100**")

        st.subheader("🎓 Recommended Courses:")
        for course in recommended_courses:
            st.markdown(f"🔗 [{course[0]}]({course[1]})")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pages = len(resume_text.split()) // 250  # Estimate pages
        user_level = "Intermediate" if pages == 2 else ("Experienced" if pages > 2 else "Fresher")
        insert_data("John Doe", "johndoe@example.com", resume_score, timestamp, pages, predicted_field, user_level, "Python, SQL", "Machine Learning, Deep Learning", ", ".join([course[0] for course in recommended_courses]))
        
        st.subheader("📺 Career Advice & Resume Tips")
        for link in random.sample(resume_videos, min(len(resume_videos), 3)):
            st.video(link)

elif choice == "Admin":
    st.header("📊 User Database")
    cursor.execute("SELECT * FROM user_data")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=["ID", "Name", "Email", "Resume Score", "Timestamp", "Pages", "Predicted Field", "User Level", "Skills", "Recommended Skills", "Recommended Courses"])
    st.dataframe(df)
    
    st.markdown("### 📥 Download User Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "resume_data.csv", "text/csv")
    
# Footer
st.markdown("---")
st.markdown("**Developed by Hadi Hijazi | AI-Powered Resume Scanner**")


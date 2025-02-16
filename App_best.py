import streamlit as st
import joblib
import pdfminer.high_level
import sqlite3
import pandas as pd
import os
import random
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

# Ensure Pafy works without youtube-dl
os.environ["PAFY_BACKEND"] = "internal"
import pafy  
import plotly.express as px  
from streamlit_tags import st_tags

# Load Pre-trained BERT Model
classifier = pipeline("text-classification", model="Rohan-Joseph/JobBERT-NER")

st.set_page_config(page_title='AI Resume Analyzer', page_icon='📄', layout='wide', initial_sidebar_state='expanded')
st.title("AI-Powered Resume Analyzer with Pre-Trained Model")
st.write("This app classifies resumes and suggests job roles using a pre-trained JobBERT model.")

# Connect to the database
connection = sqlite3.connect("resume_data.db", check_same_thread=False)
cursor = connection.cursor()

# Create a table to store user resume data
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_data (
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
    )
""")
connection.commit()

# Load category mapping (job_position_name <-> category codes)
category_mapping = pd.read_csv("category_mapping.csv", index_col=0).iloc[:, 0].to_dict()

# Extract text from PDF
def extract_text_from_pdf(file_path):
    return pdfminer.high_level.extract_text(file_path)

# Predict category using JobBERT
def predict_category(text):
    prediction = classifier(text)[0]['label']
    return category_mapping.get(prediction, "Unknown")

# Function to recommend courses based on job prediction
def get_courses(predicted_field):
    course_dict = {
        "Data Science": ds_course,
        "Web Development": web_course,
        "Android Development": android_course,
        "IOS Development": ios_course,
        "UI/UX Development": uiux_course
    }
    return course_dict.get(predicted_field, [])

# Function to calculate resume score
def calculate_resume_score(resume_text, predicted_field):
    score = 0
    resume_lower = resume_text.lower()
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
        return 50  # Default score
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_lower, field_keywords[predicted_field]])
    score = cosine_similarity(tfidf_matrix)[0, 1]
    return min(int(score * 100), 100)

# Insert data into database
def insert_data(name, email, resume_score, timestamp, pages, predicted_field, user_level, skills, recommended_skills, recommended_courses):
    cursor.execute("""
        INSERT INTO user_data (name, email, resume_score, timestamp, pages, predicted_field, user_level, skills, recommended_skills, recommended_courses)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, email, resume_score, timestamp, pages, predicted_field, user_level, skills, recommended_skills, recommended_courses))
    connection.commit()

# Streamlit Sidebar Options
st.sidebar.markdown("## User Dashboard")
activities = ['User', 'Admin']
choice = st.sidebar.selectbox("Choose Role:", activities)

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
        os.remove(temp_path)  # Cleanup

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

st.markdown("---")
st.markdown("**Developed by Hadi Hijazi | AI-Powered Resume Scanner**")

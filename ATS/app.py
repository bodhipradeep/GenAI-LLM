import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
# from dotenv import load_dotenv
import json

# Load environment variables
# load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("API_KEY"))

# Function to get response from Gemini
def get_gemini_response(resume_text, jd):
    input_prompt = f"""
    Act as a skilled ATS (Application Tracking System) evaluator. 
    Your job is to analyze resumes against job descriptions, focusing on software engineering, 
    data science, and related roles. Provide feedback in this **exact JSON structure**:
    {{
        "JD Match": "85%",
        "MissingKeywords": ["keyword1", "keyword2"],
        "Profile Summary": "Summary text"
    }}

    Resume: {resume_text}
    Job Description: {jd}

    Return only JSON data, without extra commentary or formatting.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(input_prompt)
    return response.text

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit app UIst.title("GenAI ATS")
st.title("GenAI Application Track System :blue[(ATS)]")
st.text("Know your Resume Score with GenAI ATS")

# Job description input
jd = st.text_area("Paste the Job Description", help="Provide the job description here.")

# Resume upload
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload your resume as a PDF.")

# Submit button
if st.button("Check Score"):
    if uploaded_file and jd.strip():
        # Extract text from uploaded resume
        resume_text = input_pdf_text(uploaded_file)

        # Get AI response
        response = get_gemini_response(resume_text, jd)

        # Debugging: Log raw response
        # st.write("Raw AI Response:", response)

        try:
            # Preprocess response to remove backticks or non-JSON content
            cleaned_response = response.strip().strip('```json').strip('```').strip()

            # Parse the cleaned AI response
            response_data = json.loads(cleaned_response)
            jd_match = int(response_data["JD Match"].strip('%'))
            missing_keywords = response_data["MissingKeywords"]
            profile_summary = response_data["Profile Summary"]

            # Display results
            st.subheader("Analysis Results")
            
            # Match score as circular progress graph
            st.write("#### JD Match Score")
            st.progress(jd_match / 100)
            st.write(f":blue[Match Score: **{jd_match}%**]")  # Display percentage value

            # Missing keywords as buttons
            st.write("### Missing Keywords")
            if missing_keywords:
                # Create a row of columns with a maximum of 3 buttons per row
                num_columns = 3
                cols = st.columns(num_columns)
                for i, keyword in enumerate(missing_keywords):
                    col_idx = i % num_columns  # Determine which column to place the keyword in
                    with cols[col_idx]:
                        st.markdown(f"<span style='color: orange;'>{keyword}</span>", unsafe_allow_html=True)
            else:
                st.write("No missing keywords detected!")

            # Profile summary
            st.write("### Profile Summary")
            st.text(profile_summary)

        except json.JSONDecodeError as e:
            st.error("Failed to process the response. Please try again.")
            st.write("Error Details:", str(e))
            st.write("Raw AI Response:", response)
    else:
        st.warning("Please upload a resume and provide a job description.")

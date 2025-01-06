from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini Pro Vision API and get response
def get_gemini_repsonse(input_text, image_parts, prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_text, image_parts[0], prompt])
    return response.text

# Function to handle uploaded image and convert it for Gemini
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_byte_array = io.BytesIO()
        image.save(image_byte_array, format="PNG")
        image_byte_array.seek(0)

        image_parts = [
            {
                "mime_type": "image/png",
                "data": image_byte_array.read()
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Calories Counter")
st.header("Calories Counter")
uploaded_file = st.file_uploader("Choose an image for you meal like -: Breakfast, Lunch, Dinner, Snacks etc...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me the total calories")

# Prompt for the API
input_prompt = """
You are an expert nutritionist. Analyze the food items in the image and provide:
1. Item 1 - no. of calories
2. Item 2 - no. of calories
...
"""

# Handle submission
if submit:
    try:
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_repsonse(input_prompt, image_data, input_prompt)
        st.subheader("The Response is")
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

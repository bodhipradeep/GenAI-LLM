import subprocess
import sys


# Install required libraries
def install_packages():
    required_packages = [
        "langchain",
        "langchain-community",
        "streamlit"
    ]
    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Run package installation
install_packages()


import os
import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


token = os.getenv("HF_TOKEN")  # Automatically retrieved from environment variables
if not token:
    raise ValueError("HF_TOKEN is not set. Please configure it in your environment variables or a .env file.")

llm = HuggingFaceHub(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"max_length": 128, "temperature": 0.3},
    huggingfacehub_api_token=token,
)

# Streamlit App Functions
def init_page() -> None:
    """Initializes the Streamlit page."""
    st.set_page_config(page_title="AI Chatbot")
    # Display the header
    st.header("AI Chatbot 🤖")
    # Display the subheader
    st.write("Created by Pradeep Kumar")
    st.sidebar.title("Options")

def init_messages() -> None:
    """Initializes the conversation messages."""
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant. Reply in markdown format.")
        ]

def get_answer(llm, user_input: str) -> str:
    try:
        return llm(user_input)
    except Exception as e:
        return f"An error occurred: {str(e)}"


def main() -> None:
    """Main function for the Streamlit app."""
    init_page()
    init_messages()

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing ..."):
            answer = get_answer(llm, user_input)
        st.session_state.messages.append(AIMessage(content=answer))

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

if __name__ == "__main__":
    main()

# main.py
import streamlit as st
import os
import json
from PIL import Image
import google.generativeai as genai

# Page config
st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

# Initialize Gemini
try:
    # Load config
    working_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(working_dir, "config.json")
    with open(config_file_path) as f:
        config_data = json.load(f)

    # Configure Gemini
    genai.configure(api_key=config_data["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Error initializing: {str(e)}")
    st.error("Please ensure config.json exists with your GOOGLE_API_KEY")
    st.stop()


# Gemini functions
def get_gemini_pro_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_gemini_vision_response(prompt, image):
    try:
        model = genai.GenerativeModel("gemini-pro-vision")
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_embeddings(text):
    try:
        embed = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return embed["embedding"]
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# Sidebar navigation
page = st.sidebar.radio(
    "Choose a feature",
    ["ChatBot", "Image Captioning", "Embed text", "Ask me anything"],
    index=0
)

# ChatBot Page
if page == "ChatBot":
    st.title("🤖 ChatBot")

    # Initialize chat model
    model = genai.GenerativeModel("gemini-pro")

    # Initialize chat session
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display chat history
    for message in st.session_state.chat_session.history:
        role = "assistant" if message.role == "model" else message.role
        with st.chat_message(role):
            st.markdown(message.parts[0].text)

    # Chat input
    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        try:
            response = st.session_state.chat_session.send_message(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Image Captioning Page
elif page == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file and st.button("Generate Caption"):
        try:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)

            with col2:
                caption = get_gemini_vision_response(
                    "Write a short caption for this image",
                    image
                )
                if caption:
                    st.info(caption)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Embed Text Page
elif page == "Embed text":
    st.title("🔡 Embed Text")

    text_input = st.text_area("", placeholder="Enter text to embed...")
    if st.button("Get Embeddings") and text_input:
        embeddings = get_embeddings(text_input)
        if embeddings:
            st.markdown(embeddings)

# Ask Anything Page
else:  # "Ask me anything"
    st.title("❓ Ask me a question")

    question = st.text_area("", placeholder="Ask anything...")
    if st.button("Get Answer") and question:
        response = get_gemini_pro_response(question)
        if response:
            st.markdown(response)

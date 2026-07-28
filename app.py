import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Responses for each intent
responses = {
    "greeting": "Hello! How can I help you?",
    "goodbye": "Goodbye! Have a great day!",
    "thanks": "You're welcome!",
    "name": "I am ML-Chabot, your AI assistant."
}

# Streamlit page settings
st.set_page_config(
    page_title="ML-Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 ML-Chatbot")
st.write("Welcome! Ask me anything.")

# User input
user_input = st.text_input("Type your message:")

# Predict response
if st.button("Send"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")

    else:
        # Convert text to vector
        user_vector = vectorizer.transform([user_input])

        # Predict intent
        predicted_intent = model.predict(user_vector)[0]

        # Get chatbot response
        bot_response = responses.get(
            predicted_intent,
            "Sorry, I don't understand your question."
        )

        st.success(f"🤖 {bot_response}")

import streamlit as st
from chatbot_model import get_response, pred_class, words, classes, intents_json

# Streamlit app
st.title("Chat with Our Adaptica AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")

# User input
user_input = st.text_input("You:", "")

# Handle user input
if st.button("Send"):
    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        intents_list = pred_class(user_input, words, classes)  # Ensure this returns a list
        bot_response = get_response(intents_list, intents_json)  # Pass intents_list and intents_json
        
        # Append bot response
        st.session_state.messages.append({"role": "bot", "content": bot_response})

        # Clear user input
        st.text_input("You:", "", key="user_input")


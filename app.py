import streamlit as st
from chatbot_model import get_bot_response

# Streamlit app
st.title("Chat with Our Adaptica assistant")

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
        bot_response = get_bot_response(user_input)
        
        # Append bot response
        st.session_state.messages.append({"role": "bot", "content": bot_response})

        # Clear user input
        st.text_input("You:", "", key="user_input")


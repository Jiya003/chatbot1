import streamlit as st
from chatbot_model import get_response, pred_class, words, classes, intents

# Streamlit app
st.title("Chat with Our Adaptica")

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
        intents = pred_class(user_input, words, classes)
        bot_response = get_response(intents, intents)
        
        # Append bot response
        st.session_state.messages.append({"role": "bot", "content": bot_response})

        # Clear user input
        st.text_input("You:", "", key="user_input")

# Chatbot link (optional)
st.sidebar.title("Chatbot Options")
st.sidebar.write("If you want to explore more, check out our chatbot:")
if st.sidebar.button("Open Chatbot"):
    chatbot_link = "https://your_chatbot_url"  # Replace with your actual chatbot URL
    st.sidebar.markdown(f"[Click here to interact with our chatbot]({chatbot_link})")

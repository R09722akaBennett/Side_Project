import streamlit as st
from pages.chatbot_reference.server import askAI
import threading

def get_ai_response(prompt, placeholder):
    response = askAI({"question": prompt})
    
    if response.get("status_code") == 200:
        answer = response.get("answer", "Sorry, I couldn't generate a response.")
    else:
        answer = f"An error occurred: {response.get('answer', 'Unknown error')}"
    
    placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "image": None})

def main():
    st.set_page_config(page_title="ADNEX-Chatbot Demo", layout="wide")
    st.title("💬 ADNEX-AI Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I am ADNEX Assistant. How can I help you?", "image": None}        ]

    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["image"]:
                try:
                    st.image(msg["image"], width=200)
                except Exception as e:
                    st.error(f"Error loading image: {e}")

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt, "image": None})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Start a new thread for AI response
            thread = threading.Thread(target=get_ai_response, args=(prompt, message_placeholder))
            thread.start()

if __name__ == "__main__":
    main()
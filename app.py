
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from openai import OpenAI
import streamlit as st
from chain import load_chain

st.title("ABC Bank Secure Chatbot")

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if not os.path.isdir("faiss_index"):
    st.error(
        "Knowledge base index not created \n\n"
        "Please run 'python ingest.py' to build it"
    )
    st.stop()
    
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI API KEY is not present  "
        "create and add it in .env file"
    )
    st.stop()

# validation of chain
if "chain" not in st.session_state:
    with st.spinner("starting up te assitant...."):
        st.session_state.chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []
# loading previous chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# addressing user questions
user_input = st.chat_input("Ask about accounts, credit cards, loans, FAQs, FDs..")
if user_input:
    st.session_state.messages.append({"role":"user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Thining"):
            try: 
                response = st.session_state.chain.invoke(user_input)
            except Exception as e:
                response = f"Sorry, something went wrong: {e}"
            st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    

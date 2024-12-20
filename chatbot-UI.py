import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
import json
import trust_evaluation_by_agent_team
import engagement
import politeness
import emotion

# App title
st.set_page_config(page_title="💬 AI Chatbot")

# Hugging Face Credentials
with st.sidebar:
    st.title('📍VizTrust: A Visual Analytics Tool for User Trust Development in Human-AI Communication')
    if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
        st.success('HuggingFace Login credentials already provided!', icon='✅')
        hf_email = st.secrets['EMAIL']
        hf_pass = st.secrets['PASS']
    else:
        hf_email = st.text_input('Enter E-mail:', type='default')
        # hf_email = "xinwang35314@gmail.com"
        hf_pass = st.text_input('Enter password:', type='password')
        # hf_pass = "Nogivingup1314!"
        if not (hf_email and hf_pass):
            st.warning('Please login to start!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    st.markdown(
        '🤗 Enjoy the conversation!')
    st.markdown(
        'The applied OpenAssistant LLaMA-based Models can be visited [here](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor?ref=blog.streamlit.io).'
    )
    st.markdown(
        "This is an AI chatbot prototype designed to understand user trust development in conversation with AI assistant."
    )

with st.expander("Click here for guidance"):
    st.markdown(
        "Hello! This is an AI chatbot prototype designed to understand user trust development in conversation with AI assistant. "
    )

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)


# User-provided prompt
if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, hf_email, hf_pass)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

print(st.session_state.messages)
# Ensure all messages are JSON-serializable
json_serializable_messages = [
    {"role": msg["role"], "content": str(msg["content"])}
    for msg in st.session_state.messages
]

if len(json_serializable_messages) > 1:
    start_conversation_analysis = True
else:
    start_conversation_analysis = False

# Serialize the messages to JSON
json_output = json.dumps(json_serializable_messages)

# Write the JSON output to a file
with open('data/conv.json', 'w') as f:
    f.write(json_output)

print(len(json_output))

# Start conversation analysis when getting user prompt message
if start_conversation_analysis:
    # Prepare data for visualization dashboard
    # Trust evaluation by agent team
    trust_evaluation_by_agent_team.get_trust_evalution_data(json_output)
    # User engagement data generation
    engagement.get_user_engagement_data(json_output)
    # User theory of politeness data generation
    politeness.get_politeness_data(json_output)
    # User emotion data generation
    emotion.get_emotion_data(json_output)
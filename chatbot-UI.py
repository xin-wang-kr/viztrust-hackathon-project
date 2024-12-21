import streamlit as st
from groq import Groq
import json
import trust_evaluation_by_agent_team
import engagement
import politeness
import emotion


# Groq credentials
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# App title
st.set_page_config(page_title="💬 AI Chatbot")

# Hugging Face Credentials
with st.sidebar:
    st.title('📍VizTrust: A Visual Analytics Tool for User Trust Development in Human-AI Communication')
    
    st.markdown(
        '🤗 Enjoy the conversation!')
    st.markdown(
        'This chatbot is supported by [llama-3.1-8b-instant model](https://huggingface.co/meta-llama/Llama-3.1-8B).'
    )
    st.markdown(
        "This is an AI chatbot prototype designed to understand user trust dynamics in conversation with AI assistant."
    )

with st.expander("Click here for guidance"):
    st.markdown(
        "Hello! This is an AI chatbot prototype designed to understand user trust dynamics in conversation with AI assistant. "
    )

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def generate_chat_responses(chat_completion):
    # get chat response content from the Groq API 
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# User-provided prompt

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=1000,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})


# Ensure all messages are JSON-serializable
json_serializable_messages = [
    {"role": msg["role"], "content": str(msg["content"])}
    for msg in st.session_state.messages if msg["role"] == "user"
]

# Serialize the messages to JSON
json_output = json.dumps(json_serializable_messages)

# Write the JSON output to a file
with open('data/conv.json', 'w') as f:
    f.write(json_output)


# Start conversation analysis when getting user prompt message
if len(json_serializable_messages) > 1:
    start_conversation_analysis = True
else:
    start_conversation_analysis = False

print(start_conversation_analysis)

if start_conversation_analysis:
    # Prepare data for visualization dashboard
    # Trust evaluation by agent team
    trust_evaluation_by_agent_team.get_trust_evalution_data(json_serializable_messages)
    # User engagement data generation
    engagement.get_user_engagement_data(json_serializable_messages)
    # User theory of politeness data generation
    politeness.get_politeness_data(json_serializable_messages)
    # User emotion data generation
    emotion.get_emotion_data(json_serializable_messages)
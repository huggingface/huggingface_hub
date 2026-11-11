import streamlit as st


st.title("🤖 Mein persönlicher Assistent")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.prev_id = None

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Frag mich etwas..."):
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Replace this placeholder with a call to your model or API.
    assistant_answer = f"Das ist eine Test-Antwort auf: {prompt}"

    st.session_state.history.append({"role": "assistant", "content": assistant_answer})

    with st.chat_message("assistant"):
        st.write(assistant_answer)
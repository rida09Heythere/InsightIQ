import streamlit as st
from rag import answer_question

st.set_page_config(
    page_title="InsightIQ",
    page_icon="📊",
    layout="wide"
)

st.title("InsightIQ")
st.subheader("AI Business Intelligence Copilot")

st.write("Ask questions about your uploaded business documents.")

question = st.text_input(
    "Ask a question",
    placeholder="Why did revenue decline in Q3?"
)

if st.button("Ask InsightIQ") and question.strip():
    with st.spinner("Searching documents and generating an answer..."):
        try:
            answer = answer_question(question)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
import streamlit as st

from RAG_WO_G_AI_Runtime import semantic_search as ai_search
from RAG_WO_G_ML_Runtime import semantic_search as ml_search

st.title("Semantic Question Retrieval (RAG without G)")

mode = st.radio(
    "Choose Question Bank:",
    ["AI", "ML"],
    horizontal=True
)

query = st.text_input("Enter your question:")

if query:
    if mode == "AI":
        results = ai_search(query)
    else:
        results = ml_search(query)

    if not results:
        st.warning("No results above threshold.")
    else:
        for score, text in results:
            st.markdown(f"**Score:** {score:.3f}")
            st.write(text)
            st.markdown("---")

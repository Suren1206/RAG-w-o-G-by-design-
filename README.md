# RAG-w-o-G-by-design

This is a customized Python tool built for a specific purpose. As the name suggests, it is a RAG-like utility that sits between a full-fledged RAG system and a purely manual search.

The tool is intended for students who want to refresh and deepen their understanding by revisiting past questions from earlier tests. It enables semantic search over a question bank, allowing users to retrieve similar questions one at a time while working on new ones.

## Motivation & Context

I felt the need for this tool during my academic journey over the past few months while studying AI and ML in a structured manner through certification courses. As part of this process, I regularly took weekly tests using ChatGPT, consisting of MCQs, conceptual questions, and scenario-based problems.

Over time, I began noticing subtle variations and deeper probing of the same core concepts across different test papers. Revisiting these semantically similar questions turned each weekly test into a richer learning experience. Probing the evaluation logic of LLMs and challenging their feedback also added a valuable meta-learning layer.

While this was manageable initially, after 8–10 weeks I realized the opportunity to systematically revisit past questions, attempt them with greater precision, and improve answer quality.

I personally observed two advantages from this approach:

Understanding is a spiral process — insights deepen when we return to similar topics periodically

As knowledge genuinely expands, dependence on the question bank naturally reduces

This self-help tool has been extremely useful for me. However, it works best when its purpose is clearly understood and when it is used with discipline — as a support mechanism, not a crutch.

Sharing with delight.

## Setup & Run

### Prerequisites
- Python 3.9+
- OpenAI API key available as an environment variable

## Setup & Run

### Prerequisites
- Python 3.9+
- OpenAI API key available as an environment variable

### Installation
```bash
pip install -r requirements.txt
```
### Run
```bash
streamlit run RAG_WO_G_Streamlit.py
```
---

## How to Use the Tool

### Quick Start (No Customization)

1. Download or clone the entire repository locally  
2. Run `RAG_WO_G_Streamlit.py`  
3. Select **AI** or **ML** question bank  
4. Enter a query to retrieve semantically similar questions  

---

### Customization (Using Your Own Question Bank)

1. Replace the contents of the respective Question Bank files  
2. Ensure each question ends with `?` or `.`  
3. Run the rebuild scripts:
   - `RAG_WO_G_AI_Rebuild.py`
   - `RAG_WO_G_ML_Rebuild.py`  
   *(This regenerates the FAISS index and stored texts)*

4. **Do NOT modify the `*_Runtime.py` files**  
   These are read-only runtime loaders used by Streamlit

5. Run `RAG_WO_G_Streamlit.py` again to query your customized data

---

### Optional Reading
For detailed design philosophy and architectural decisions, see `DESIGN_DETAILS.md`.


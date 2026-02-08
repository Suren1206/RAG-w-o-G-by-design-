# RAG-w-o-G-by-design : Design Details

## Specific configuration

LLM: gpt-4o-mini

Embedding model: text-embedding-3-large

Vector DB: FAISS

## Design Philosophy

In classic RAG systems, the LLM becomes the final authority — it decides what matters and synthesizes answers. This can introduce risks such as hallucination, over-summarization, or loss of original context.

In this tool, the human remains the final authority.
Nothing is rewritten, nothing is synthesized, and nothing is hidden.
The “G” (Generation) step is not forgotten — it is deliberately excluded by design.

## Retrieval Without Generation

This tool implements Retrieval without Generation, with only a very limited and controlled use of LLM generation.

The retrieval pipeline — embeddings, vector search, and semantic similarity — follows standard RAG practices. The LLM is used only when unavoidable, i.e., to summarize a question if its token length exceeds a predefined limit. This ensures that vectorization remains feasible without altering the original intent of the question.

From a cost perspective, this approach is highly economical, as it avoids unnecessary LLM generation.

## Question Bank Processing

The original Question Bank file contains raw data and undergoes the following preprocessing steps:

Normalization and cleanup

Removal of MCQ options

Use of ? or . as delimiters to identify question boundaries

Since the entire question bank was generated earlier using LLMs at different points in time, the data is already well-structured and disciplined, requiring no manual intervention.

## Vectorization & Similarity Logic

Token-based vectorization is applied with a maximum of 500 tokens

A similarity threshold check of 0.7 is used, with an additional attempt to regenerate the vector if required

If regeneration fails, the original text is preserved and stored

No rewriting or skipping of original content is possible under this configuration

During experimentation, it was observed that retrieving all questions with cosine similarity ≥ 0.625 provides a good balance. This reliably surfaces related historical questions that meaningfully support concept recall and reinforcement.

## Streamlit Access Model

A dedicated runtime file is created exclusively for Streamlit access. This allows the UI to interact with the existing vector database directly, without relying on prompts.

The Streamlit interface provides:

Selection between AI and ML question banks

Read-only semantic search over the existing database

## Current Scope Limitation

The baseline model does not support incremental data updates.
This is an intentional design decision. Incremental handling will be introduced later as a separate, well-isolated module.

## Enhancement Plans & Future Possibilities

Incremental data handling module for faster updates and optimized maintenance cost

Progress logging and improved input/output validation checks

Score introspection utilities (e.g., understanding why a query scored 0.56 vs 0.82)

Comparative study of LLM behavior across different categories of questions

Impact analysis of implementing the same logic in orchestrated environments (e.g., LangChain)

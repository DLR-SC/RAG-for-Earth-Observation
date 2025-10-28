# DARES25

This guide explains how to reproduce the evaluation presented in the **DARES25** paper **Knowledge Graph-Enhanced Retrieval-Augmented
Generation for Earth Observation Data**. The code in his repository can be used to run the evaluation. To learn how to get and run the used version of the graph, please refer to the instructions given in [🧠 Database Setup (ArangoDB)](README.md#-database-setup-arangodb).

\* Creating the graph from scratch is currently not possible as the needed TaxoTagger ist not public software.

## Question and Answer Set

The generated questions (and respectiva answers) that were evaluated can be found on zenodo at [A Dataset for Earth Observation Question–Answering Using a RAG-Based Model](<https://zenodo.org/records/17287798>) or `evaluation/prompt_generator` / `evaluation/DARES25/answers_w_structured_context.csv`. If you do want to generate the answers from scratch you have to start the FastAPI server and call the URL. The simplest way to get everything runnig is to start the whole application (including the frontend) as detailled in the [README](README.md#-installation). Alternatively you can start just the backend with the provided Dockerfile at `backend/Dockerfile`.

Simple RAG answers are unavailable via the API and must be generated using the methods in `backend/app/rag.py`, e.g. `generate_sample`.

## Evlauating the Answers

The code used to evaluate the answers can be found at `evaluation/DARES25`.

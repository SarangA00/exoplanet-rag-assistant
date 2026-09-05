# Exoplanet Analysis & Local AI Q&A Assistant

An end-to-end pipeline that analyzes NASA's Planetary Systems Composite
Parameters (PSCompPars) exoplanet catalog and exposes the results through a
locally-run, retrieval-augmented generation (RAG) question-answering system.
The project spans data preparation, three independent statistical analyses,
and a full RAG architecture (embedding, vector search, grounded generation)
built entirely on open-source tooling and running on local compute.

Built by Sarang Athani during a summer internship at NoduAI, as a
self-directed applied-AI project.

## What it does

1. **Prepares the data** — splits the NASA exoplanet catalog into an 80/20
   train/test set with a fixed random seed, so all analysis is developed and
   tuned on the training set only.
2. **Explores the data** — completeness, distribution, and correlation
   analysis across confirmed exoplanets and their host stars.
3. **Models relationships** — fits a mass-radius power law
   (R ∝ M^0.355, R² = 0.84) and tests whether host-star temperature, radius,
   and mass predict planet size (R² = 0.20 — they're weak predictors).
4. **Classifies habitability** — sorts every planet into Too Hot / Habitable
   Zone / Too Cold / Unknown using Kopparapu et al. (2013) insolation-flux
   thresholds, then compares the breakdown across detection methods.
5. **Answers questions about the findings** — a 4-stage local RAG pipeline
   (findings generation → embedding → vector indexing → grounded
   retrieval/generation) lets you ask questions in plain English and get
   answers grounded only in the verified analysis output — with the
   assistant abstaining rather than guessing when nothing relevant is found.

## Key findings

- Only ~3% of known exoplanets fall in the habitable zone.
- Detection method strongly biases what we find: transit-discovered planets
  are classified "Too Hot" 94% of the time, vs. 60% for radial-velocity
  discoveries — a clear selection-bias signal.
- Planet radius and mass are strongly correlated (|r| = 0.92) and follow a
  single power law until the transition to gas giants.
- Host-star properties alone are weak predictors of planet size — other
  factors (like orbital distance) likely dominate.

## Tech stack

- **Analysis:** Python, pandas, NumPy, scikit-learn, matplotlib/seaborn
- **RAG system:** sentence-transformers (local embeddings), sqlite-vec
  (vector search), Ollama running Llama 3.2 3B (local generation),
  pdfplumber (PDF extraction), Streamlit (demo UI)

## Repo structure

```
eda_analysis.py          # Module 1: exploratory data analysis
regression_analysis.py   # Module 2: mass-radius & stellar regressions
hz_classification.py     # Module 3: habitable zone classification
split_dataset.py         # train/test split (run first)
*_output/                # generated plots, CSVs, JSON summaries
rag/                     # local RAG pipeline (embedding, indexing, Q&A, Streamlit app)
Project_Summary_Report.docx   # plain-language project summary
Technical_Report.docx         # full technical writeup with methodology
```

## Running it

```bash
# 1. Split the raw NASA catalog into train/test sets
python split_dataset.py

# 2. Run each analysis module
python eda_analysis.py
python regression_analysis.py
python hz_classification.py

# 3. Build and query the RAG assistant
cd rag
pip install -r requirements.txt
python build_findings.py   # turn analysis JSON into readable findings text
python build_index.py      # embed + index findings and papers
streamlit run app.py       # launch the Q&A UI
```

This project runs entirely locally — no data or queries are sent to any
external API.


import streamlit as st

from ask import DEFAULT_K, DISTANCE_THRESHOLD, answer_from_results, get_connection, search
from embedder import Embedder

st.set_page_config(page_title="Exoplanet RAG Demo", page_icon="🔭")


@st.cache_resource
def load_embedder() -> Embedder:
    return Embedder()


st.title("Exoplanet RAG Demo")
st.caption("Ask a question about the exoplanet findings and papers, grounded by local retrieval + Ollama.")

with st.sidebar:
    k = st.slider("k (results to retrieve)", min_value=1, max_value=5, value=DEFAULT_K)
    threshold = st.slider(
        "Distance threshold", min_value=0.5, max_value=2.0, value=DISTANCE_THRESHOLD, step=0.05
    )

question = st.text_input("Your question")
submitted = st.button("Submit")

if submitted and question.strip():
    embedder = load_embedder()
    query_vector = embedder.embed_one(question)

    db = get_connection()
    results = search(db, query_vector, k)
    db.close()

    st.subheader("Retrieved chunks")
    for r in results:
        over = r["distance"] > threshold
        flag = "  (over threshold, ignored)" if over else ""
        label = f"[{r['source_type']}:{r['source_title']} chunk {r['chunk_index']}] distance={r['distance']:.4f}{flag}"
        with st.expander(label):
            st.write(r["text"])

    st.subheader("Answer")
    with st.spinner("Generating answer..."):
        answer = answer_from_results(question, results, threshold)
    st.markdown(answer)
elif submitted:
    st.warning("Please enter a question.")

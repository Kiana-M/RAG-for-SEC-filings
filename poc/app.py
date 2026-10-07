"""Streamlit demo: streamlit run app.py"""
import streamlit as st

from answer import answer

EXAMPLES = [
    "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
    "How has NVIDIA's revenue and growth outlook changed over the last two years?",
    "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
]

st.set_page_config(page_title="SEC Filings Q&A", layout="wide")
st.title("SEC filings Q&A")
st.caption("Answers from 245 10-K / 10-Q filings of 54 companies, with citations to the filing excerpts used.")

example = st.selectbox("Example questions", [""] + EXAMPLES)
question = st.text_area("Question", value=example, height=80)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Planning, searching and writing the answer..."):
        try:
            st.session_state["out"] = answer(question.strip())
        except Exception as e:  # quota exhaustion, network errors
            st.session_state.pop("out", None)
            st.error(f"{type(e).__name__}: {e}")

out = st.session_state.get("out")
if out:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Answer")
        st.caption(f"Answer generated in {out['answer_llm_calls']} LLM call(s) from {sum(len(c) for c in out['retrieved'].values())} retrieved excerpts.")
        st.markdown(out["answer"])
    with right:
        st.subheader("Gaps")
        if out["gaps"]:
            for g in out["gaps"]:
                st.warning(g)
        else:
            st.success("No coverage gaps for the requested companies and periods.")
        for n in out["notes"]:
            st.info(n)
        if out["invalid_citations"]:
            st.error("Cited ids not in the retrieved set: " + ", ".join(out["invalid_citations"]))
        st.subheader("Plan")
        st.json(out["plan"], expanded=False)

    st.subheader(f"Citations ({len(out['citations'])})")
    for c in out["citations"]:
        m = c["meta"]
        with st.expander(f"[{c['id']}]  {m['company']} · {m['form']} {m['fiscal_label']} · {m['section']}"):
            st.caption(f"Period ending {m['period_end']} · filed {m['filing_date']} · {m['file']} · "
                       f"{m['sector']} / {m['industry']} · RRF {c['rrf']} (dense #{c['dense_rank']}, BM25 #{c['bm25_rank']})")
            st.text(c["text"])

    if out["summaries"]:
        with st.expander("Per-company summaries"):
            for t, s in out["summaries"].items():
                st.markdown(f"**{t}**\n\n{s}")

import streamlit as st
from newspaper import Article
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_summarizer():
    return pipeline("summarization")

@st.cache(allow_output_mutation=True)
def load_qa_pipeline():
    return pipeline("question-answering")

summarizer = load_summarizer()
qa_pipeline = load_qa_pipeline()

def fetch_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error fetching article: {e}")
        return None

def summarize_article(article_text):
    try:
        summary = summarizer(article_text, max_length=150, min_length=100, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error summarizing article: {e}")
        return None

def answer_question(article_text, question):
    try:
        result = qa_pipeline(question=question, context=article_text)
        return result['answer']
    except Exception as e:
        st.error(f"Error answering question: {e}")
        return None

def run_app():
    st.title("Newsbot: Automated News Summarizer and QnA")

    url = st.text_input("Enter News Article URL")
    if 'article_text' not in st.session_state:
        st.session_state.article_text = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None

    if st.button("Fetch Article"):
        if url:
            st.session_state.article_text = fetch_article(url)
            if st.session_state.article_text:
                st.success("Article fetched successfully")
            else:
                st.error("Failed to fetch article.")
        else:
            st.error("Please enter a valid URL")

    if st.session_state.article_text:
        if st.button("Summarize Article"):
            st.session_state.summary = summarize_article(st.session_state.article_text)
            if st.session_state.summary:
                st.subheader("Summary")
                st.write(st.session_state.summary)
            else:
                st.error("Failed to summarize article.")

    if st.session_state.summary:
        question = st.text_input("Enter Your Question")
        if st.button("Answer Your Question"):
            if question:
                answer = answer_question(st.session_state.article_text, question)
                if answer:
                    st.subheader("Answer to Your Question")
                    st.write(answer)
                else:
                    st.error("Failed to answer question.")
            else:
                st.error("Please enter a question")

if __name__ == "__main__":
    run_app()

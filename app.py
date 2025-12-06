import streamlit as st
import os
from src.rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="Financial ML Q&A Bot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline (cached)."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        return RAGPipeline(api_key=api_key)
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {str(e)}")
        st.stop()


def main():
    # Header
    st.markdown('<div class="main-header">📚 Financial ML Q&A Bot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask questions about "Advances in Financial Machine Learning" by Marcos Lopez de Prado</div>',
        unsafe_allow_html=True
    )
    
    # Load pipeline
    with st.spinner("Loading AI model..."):
        rag = load_rag_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This bot answers questions based on the book:
        **"Advances in Financial Machine Learning"**
        by Marcos Lopez de Prado
        """)
        
        st.divider()
        
        st.header("Settings")
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources = more context but slower responses"
        )
        
        show_sources = st.checkbox("Show source excerpts", value=True)
        
        st.divider()
        
        st.header("Example Questions")
        example_questions = [
            "What is the triple barrier method?",
            "How does cross-validation work in financial ML?",
            "What is meta-labeling?",
            "Explain the concept of sample weights",
            "What are ensemble methods in financial ML?",
        ]
        
        for question in example_questions:
            if st.button(question, key=question):
                st.session_state.example_question = question
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i} - Page {source['page']}** (Similarity: {source['similarity']:.3f})")
                        if show_sources:
                            st.markdown(f"```\n{source['text']}\n```")
    
    # Handle example question
    if "example_question" in st.session_state:
        user_input = st.session_state.example_question
        del st.session_state.example_question
    else:
        user_input = st.chat_input("Ask a question about the book...")
    
    # Process user input
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = rag.query(user_input, top_k=top_k, return_sources=True)
                answer = result["answer"]
                sources = result["sources"]
                
                st.markdown(answer)
                
                with st.expander("View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i} - Page {source['page']}** (Similarity: {source['similarity']:.3f})")
                        if show_sources:
                            st.markdown(f"```\n{source['text']}\n```")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
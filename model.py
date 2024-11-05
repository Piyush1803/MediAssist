import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import os

DB_FAISS_PATH = 'db'

# Custom prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """Prompt template for QA retrieval."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Define Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Load the LLM
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    # Load embeddings
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # Allow deserialization of the FAISS index if it's a trusted file
    print(f"Loading FAISS index from {DB_FAISS_PATH}...")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Debugging: Check if the database is loaded
    if db is not None:
        print("FAISS index loaded successfully.")
    else:
        print("Failed to load FAISS index.")

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    
    return qa

# Streamlit UI
st.title("Medical Bot")

# Initialize the bot once in session state
if 'qa' not in st.session_state:
    st.session_state.qa = qa_bot()
    st.write("QA bot initialized.")

# User input
query = st.text_input("Enter your query:")

if query:
    # Debugging: Show the query received
    st.write(f"Query received: {query}")
    
    # Get the response from the bot
    response = st.session_state.qa({'query': query})

    # Debugging: Print the raw response
    st.write("Raw response:", response)

    answer = response["result"]

    # Add sources if available
    if "source_documents" in response and response["source_documents"]:
        answer += f"\n\nSources: {response['source_documents']}"
        # Debugging: Show source documents
        st.write("Source documents:", response['source_documents'])
    else:
        answer += "\n\nNo sources found"
    
    # Display the answer
    st.write(answer)

    # Debugging: Show final answer
    st.write("Final answer displayed:", answer)

import os
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTOR_DB_PATH = "faiss_index_constitution"  
MODEL_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf" 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_chain():
    """
    Loads an existing vector database and creates a conversational retrieval chain
    with custom prompts and memory.

    Returns:
        ConversationalRetrievalChain: The configured LangChain QA chain, or None if setup fails.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"❌ Error: Vector DB not found at '{VECTOR_DB_PATH}'")
        print("Please run your ingestion script first to create the vector database.")
        return None

    print("🚀 Initializing chatbot...")


    print(f"   - Loading embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


    print(f"   - Loading vector database from '{VECTOR_DB_PATH}'...")
    db = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True  # Required for FAISS loading
    )
    print("   ✅ Vector database loaded.")

    print(f"   - Loading LLM from '{MODEL_PATH}'...")
    llm = LlamaCpp(
        streaming=True,
        model_path=MODEL_PATH,
        temperature=0.75,
        top_p=1,
        verbose=False, # Set to True for detailed LLM output
        n_ctx=32768,
    )
    print("   ✅ LLM loaded.")

    
    # This prompt helps rephrase a follow-up question into a standalone one
    condense_question_template = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(condense_question_template)

    # This prompt structures how the LLM should answer using the retrieved documents
    answer_generation_template = """
    Use the following pieces of information from the document to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_template(answer_generation_template)


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    
    retriever = db.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 relevant chunks

    # Final Conversational Retrieval Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": ANSWER_GENERATION_PROMPT},
    )

    print("\n✅ Chatbot is ready to answer your questions!")
    return chain

def run_chat_interface(chain):
    """
    Runs the main command-line interface loop for the chatbot.
    """
    print("--- Document Q&A Chatbot ---")
    print("Ask any question about your document. Type 'exit' or 'quit' to end.")

    while True:
        try:
            query = input("\nYour Question: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting chatbot. Goodbye! 👋")
                break
            if not query.strip():
                continue

            print("\n🤔 Thinking...")

            # The chain will automatically use the memory to handle chat history
            result = chain.invoke({"question": query})
            
            print("\n--- Answer ---")
            print(result["answer"])

            
            if result.get("source_documents"):
                print("\n--- Sources ---")
                for i, doc in enumerate(result["source_documents"]):
                    page_num = doc.metadata.get('page', 'N/A')
                    print(f"Source {i+1} (Page {page_num}):")
                    print(f'   "{doc.page_content[:250].strip().replace(chr(10), " ")}..."\n')

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

def main():
    """
    Main function to initialize and run the chatbot.
    """
    qa_chain = create_chain()
    if qa_chain:
        run_chat_interface(qa_chain)

if __name__ == "__main__":
    main()
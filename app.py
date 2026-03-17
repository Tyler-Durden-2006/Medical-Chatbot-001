from flask import Flask, render_template, request, jsonify
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv
from pinecone import Pinecone
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Initialize retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize chat model
chatModel = ChatOllama(
    model="phi3:mini",
    temperature=0.2
)

# System prompt
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Format docs function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain
rag_chain = (
    {
        "context": itemgetter("input") | retriever | format_docs,
        "input": itemgetter("input")
    }
    | prompt
    | chatModel
    | StrOutputParser()
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def get_response():
    try:
        user_message = request.form.get('msg', '').strip()
        if not user_message:
            return jsonify({"response": "Please enter a message."})
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": user_message})
        return response
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
🏥 Medical Chatbot (RAG + LLM + Ollama)

An AI-powered medical chatbot that answers user queries using a medical knowledge base.
This project uses Retrieval-Augmented Generation (RAG) with a local LLM (via Ollama) to provide accurate and context-aware responses.



🚀 Features
	•	💬 Chat-based interface for medical queries
	•	📚 Uses a medical PDF as knowledge base
	•	🧠 Retrieval-Augmented Generation (RAG) pipeline
	•	⚡ Runs locally using Ollama (no API cost)
	•	🌐 Simple web UI using Flask
	•	🔍 Context-aware answers using embeddings

⚙️ How It Works
	1.	PDF is processed and converted into embeddings
	2.	Embeddings are stored in a vector database
	3.	User asks a question
	4.	Relevant context is retrieved
	5.	LLM generates an answer using that context


  💡 Usage
	•	Ask any medical-related question
	•	The chatbot retrieves relevant info from the PDF
	•	Generates a contextual response using LLM

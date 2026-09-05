🤖 Personalized LangChain Chatbot & RAG Learning Project

A hands-on Generative AI learning project that demonstrates how to build a conversational chatbot with LangChain, Groq, message history, prompt templates, conversation trimming, vector stores, retrievers, and Retrieval-Augmented Generation (RAG).

This project is designed as a practical learning journey through the core building blocks required for modern LLM applications.

📌 Project Overview

The project contains two Jupyter notebooks:

Notebook

Purpose

chatbot.ipynb

Builds a personalized conversational AI chatbot with Groq and LangChain

vectorretriever.ipynb

Demonstrates documents, embeddings, Chroma vector stores, retrievers, and a basic RAG pipeline

The chatbot is personalized around the learning journey of Ayush Kumar Verma, a student exploring:

Artificial Intelligence

Machine Learning

Generative AI

Python

LangChain

Large Language Models (LLMs)

AI Application Development

✨ Features

1. Groq LLM Integration

The project connects LangChain with Groq using ChatGroq.

from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

2. Environment Variable Management

API keys are loaded using a .env file:

from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

3. Available Groq Model Discovery

The chatbot notebook includes code to query the Groq API and display models available to the current API key.

url = "https://api.groq.com/openai/v1/models"

response = requests.get(
    url,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
)

for model in response.json()["data"]:
    print(model["id"])

4. Basic Conversational Chatbot

The project uses LangChain message objects:

HumanMessage

AIMessage

SystemMessage

Example:

messages = [
    HumanMessage(
        content="Hi, my name is Ayush Kumar Verma. I am learning AI and Machine Learning."
    )
]

response = model.invoke(messages)
print(response.content)

5. Conversation Memory

The chatbot uses RunnableWithMessageHistory to maintain separate conversation sessions.

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

A dictionary-based in-memory store manages chat sessions:

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

6. Session-Based Conversations

config = {
    "configurable": {
        "session_id": "chat1"
    }
}

Different session IDs create independent conversation histories:

chat1 → conversation 1
chat2 → conversation 2
chat3 → conversation 3

7. Personalized System Prompts

The project uses ChatPromptTemplate and MessagesPlaceholder to provide custom instructions.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intelligent and helpful AI assistant.

        The primary user in this learning project is Ayush Kumar Verma.
        Ayush is a student interested in Artificial Intelligence,
        Machine Learning, and Generative AI.

        Be clear, practical, and educational.
        Explain concepts step-by-step when needed.
        """
    ),
    MessagesPlaceholder(variable_name="messages")
])

8. Multi-Language Prompt Support

The chatbot can dynamically change its response language.

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

        Answer clearly and educationally in {language}.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
])

Example:

chain.invoke({
    "messages": [
        HumanMessage(content="Explain Artificial Intelligence")
    ],
    "language": "Hindi"
})

9. Conversation History Management

Long conversations can consume a large number of tokens.

This project uses trim_messages() to limit the message history sent to the model.

from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

🧠 Vector Stores and Retrievers

The second notebook explores the foundation of Retrieval-Augmented Generation (RAG).

Documents

LangChain documents contain:

page_content

metadata

Example:

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"}
    )
]

The project creates sample documents about:

Dogs

Cats

Goldfish

Parrots

Rabbits

🔢 Embeddings

The project uses Hugging Face embeddings:

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

Embeddings convert text into numerical vector representations that can be compared semantically.

🗄️ Chroma Vector Database

from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings
)

🔍 Similarity Search

vectorstore.similarity_search("cat")

The notebook also demonstrates:

vectorstore.similarity_search_with_score("cat")

and asynchronous retrieval:

await vectorstore.asimilarity_search("cat")

🔎 Retrievers

A retriever acts as the bridge between the vector database and an LLM application.

Custom Runnable Retriever

from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(
    vectorstore.similarity_search
).bind(k=1)

Standard VectorStore Retriever

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

🚀 Retrieval-Augmented Generation (RAG)

The project builds a simple RAG pipeline using LangChain Expression Language (LCEL).

User Question
      │
      ▼
Retriever
      │
      ▼
Relevant Documents
      │
      ▼
Prompt Template
      │
      ▼
LLM
      │
      ▼
Final Answer

RAG Chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

Question:
{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("human", message)
])

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

response = rag_chain.invoke("Tell me about dogs")

print(response.content)

🛠️ Technologies Used

Technology

Purpose

Python

Main programming language

LangChain

LLM application framework

Groq

High-speed LLM inference

ChatGroq

LangChain integration for Groq

Hugging Face

Embedding models

Sentence Transformers

Text embedding generation

Chroma

Vector database

python-dotenv

Environment variable management

Transformers

Tokenization support

Tiktoken

Token counting utilities

Jupyter Notebook

Interactive development environment

📁 Project Structure

project-folder/
│
├── chatbot.ipynb
│   ├── Groq API integration
│   ├── LLM initialization
│   ├── HumanMessage and AIMessage
│   ├── Chat history
│   ├── Session management
│   ├── Prompt templates
│   ├── Personalized system prompts
│   ├── Multi-language responses
│   └── Conversation trimming
│
├── vectorretriever.ipynb
│   ├── LangChain Documents
│   ├── Hugging Face Embeddings
│   ├── Chroma Vector Store
│   ├── Similarity Search
│   ├── Async Search
│   ├── Retrievers
│   └── RAG Pipeline
│
├── .env
├── requirements.txt
└── README.md

⚙️ Installation

1. Clone the Repository

git clone <your-repository-url>
cd <your-project-folder>

2. Create a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

macOS/Linux

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

If you do not have a requirements.txt file yet:

pip install langchain langchain-core langchain-groq langchain-community langchain-chroma langchain-huggingface sentence-transformers transformers tiktoken python-dotenv requests jupyter

🔐 Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
LANGCHAIN_API_KEY=your_langchain_api_key_here

Optional LangSmith configuration:

LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langchain-chatbot-project

⚠️ Important Security Rule

Never upload your real .env file or API keys to GitHub.

Add this to .gitignore:

.env
venv/
__pycache__/
.ipynb_checkpoints/

If an API key was previously exposed publicly, revoke or rotate it immediately.

📦 Recommended requirements.txt

langchain
langchain-core
langchain-groq
langchain-community
langchain-chroma
langchain-huggingface
sentence-transformers
transformers
tiktoken
python-dotenv
requests
jupyter

▶️ How to Run

Start Jupyter

jupyter notebook

or:

jupyter lab

Then run the notebooks in this order:

Step 1: Run chatbot.ipynb

This notebook covers:

Loading the Groq API key

Checking available Groq models

Initializing the LLM

Sending messages to the model

Creating conversation history

Managing multiple sessions

Using prompt templates

Adding system instructions

Managing language preferences

Trimming conversation history

Step 2: Run vectorretriever.ipynb

This notebook covers:

Creating documents

Generating embeddings

Creating a Chroma vector store

Running similarity searches

Creating retrievers

Building a RAG pipeline

🧩 LangChain Concepts Covered

ChatGroq

HumanMessage

AIMessage

SystemMessage

ChatMessageHistory

RunnableWithMessageHistory

ChatPromptTemplate

MessagesPlaceholder

RunnablePassthrough

RunnableLambda

trim_messages

Document

HuggingFaceEmbeddings

Chroma

VectorStore Retrievers

LCEL (LangChain Expression Language)

Retrieval-Augmented Generation (RAG)

🏗️ Project Architecture

Chatbot Architecture

User Input
    ↓
Prompt Template
    ↓
Message History
    ↓
Message Trimming
    ↓
Groq LLM
    ↓
AI Response

RAG Architecture

User Question
    ↓
Retriever
    ↓
Chroma Vector Database
    ↓
Relevant Documents
    ↓
Prompt + Context
    ↓
LLM
    ↓
Final Response

🐛 Common Issues and Solutions

GROQ_API_KEY not found

Make sure your .env file contains:

GROQ_API_KEY=your_key_here

Then restart the notebook kernel.

Model Not Found

Do not assume every Groq model is available to every account.

Check available models:

import requests

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
)

for model in response.json()["data"]:
    print(model["id"])

Use one of the model IDs returned by your account.

Note: The vector retriever notebook contains an older model identifier (Llama3-8b-8192). If it is unavailable for your Groq account, replace it with a model returned by the /models endpoint, such as openai/gpt-oss-20b if available.

Could Not Import transformers

Install:

%pip install transformers tiktoken

Then restart the Jupyter kernel.

Invalid Input Type HumanMessage

Pass a list of messages when invoking a LangChain chat model directly:

response = model.invoke([
    HumanMessage(content="Hello")
])

🎯 Learning Outcomes

After completing this project, you should understand how to:

Connect an LLM API with LangChain

Build a basic AI chatbot

Pass structured messages to an LLM

Maintain conversation memory

Manage multiple chat sessions

Use system prompts and prompt templates

Control conversation history size

Generate text embeddings

Store embeddings in a vector database

Perform semantic similarity search

Build retrievers

Create a basic RAG application using LCEL

🚧 Future Improvements

Persistent chat memory using Redis or a database

Persistent Chroma vector database

PDF and document ingestion

Text chunking with RecursiveCharacterTextSplitter

Conversational RAG

Source citations in RAG responses

Streaming responses

Streamlit or FastAPI web interface

User authentication

Chat history database

LangSmith tracing and evaluation

Agent and tool integration

Multi-document knowledge base

Production deployment

👨‍💻 Author

Ayush Kumar Verma

Student and AI enthusiast currently exploring:

Artificial Intelligence • Machine Learning • Generative AI • LangChain • LLM Applications

This repository documents a practical learning journey toward building real-world AI applications.

📚 Project Status

🟢 Learning / Development Project

This project focuses on understanding LangChain fundamentals before moving toward advanced applications such as:

Conversational RAG

AI Agents

Tool Calling

Multi-Agent Systems

Production AI Applications

⭐ If You Found This Project Useful

Consider giving the repository a ⭐ on GitHub and following the journey as the project evolves.

Built with ❤️ using Python, LangChain, Groq, Hugging Face, Chroma, and Generative AI.

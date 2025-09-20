# rag_pipeline.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import getpass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# LANGSMITH_API_KEY=os.environ["LANGSMITH_API_KEY"]
# os.environ["LANGSMITH_TRACING"] = "true"


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


class RagPipeline:
    def __init__(self):
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.documents = []
        self.llm = llm
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        # Create an empty vector store if it doesn't exist
        placeholder_text = ["Initialize with some text"]
        self.vector_store = FAISS.from_texts(placeholder_text, self.embeddings)

    def _load_documents(self, filepath):
        _, extension = os.path.splitext(filepath)
        if extension.lower() == '.pdf':
            loader = PyPDFLoader(filepath)
        else: # Default to text loader
            loader = TextLoader(filepath, encoding='utf-8')
        return loader.load()

    def add_document(self, filepath):
        """Loads and processes a single file, adding it to the vector store."""
        print(f"Processing document: {filepath}")
        try:
            new_docs = self._load_documents(filepath)
            chunks = self.text_splitter.split_documents(new_docs)
            if chunks:
                self.vector_store.add_documents(chunks)
                print(f"Added {len(chunks)} chunks from {filepath} to the knowledge base.")
            else:
                print(f"Warning: No text chunks extracted from {filepath}.")
        except Exception as e:
            print(f"Error processing document {filepath}: {e}")

    def add_new_pathway_documents(self, pathway_data):
        """Adds documents coming from the Pathway live stream."""
        print(f"Processing {len(pathway_data)} documents from Pathway live stream...")
        docs_to_add = []
        for item in pathway_data:
            # Create LangChain Document objects from Pathway's output
            from langchain.schema import Document
            # Assuming item is a dictionary-like object from Pathway
            # with 'doc' and 'metadata' fields
            doc_content = item.get('doc', '')
            metadata = {'source': item.get('metadata', 'live_stream')}
            docs_to_add.append(Document(page_content=doc_content, metadata=metadata))
        
        if docs_to_add:
            chunks = self.text_splitter.split_documents(docs_to_add)
            if chunks:
                self.vector_store.add_documents(chunks)
                print(f"Added {len(chunks)} chunks from Pathway to the knowledge base.")


    def generate_report(self, question):
        """Generates a report by retrieving context, building a prompt, and calling the LLM chain."""
        print(f"Generating report for question: '{question}'")
        try:
            # 1. Retrieve relevant documents (same as before)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4}) # Increased to 4 for more context
            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                return {"summary": "Could not find relevant information in the uploaded documents to answer the question.", "sources": []}
            
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
            
            # 2. Build the Prompt and Chain
            
            system_message = """
            You are a smart research assistant. Your task is to answer a user's question based *only* on the provided context.
            
            Follow these rules strictly:
            1.  Synthesize the information from the context into a concise, structured report.
            2.  Do not use any information outside of the provided context.
            3.  If the context does not contain the answer, you must state that you cannot answer the question with the given information.
            4.  Do not make up facts or speculate.
            """
            
            # The human message template formats the inputs for the LLM.
            human_message = """
            Context:
            {context}
            ---
            Question: {question}
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", human_message)
            ])
            
            # The output parser cleans up the LLM's response into a simple string.
            output_parser = StrOutputParser()
            
            # The chain connects the prompt, the LLM, and the output parser.
            chain = prompt | self.llm | output_parser
            
            # 3. Invoke the chain with the context and question
            summary = chain.invoke({
                "context": context,
                "question": question
            })
            
            # 4. Return the result (same as before)
            return {
                "success": True,
                "summary": summary,
                "takeaways": [ # In a real app, you might ask the LLM to generate these too!
                    "This is a key takeaway generated by the AI.",
                    "The system identified critical patterns in the data.",
                ],
                "sources": sources
            }
        except Exception as e:
            print(f"Error during report generation: {e}")
            return {"error": "Failed to generate report."}
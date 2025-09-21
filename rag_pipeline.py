import os
import getpass
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# This llm object is now created here to be passed into the pipeline
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # Adding JSON mode for more reliable analysis
    model_kwargs={"response_format": {"type": "json_object"}},
)

class RagPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        # Initialize with placeholder text to create the vector store index
        placeholder_text = ["This is a placeholder to initialize the FAISS vector store."]
        self.vector_store = FAISS.from_texts(placeholder_text, self.embeddings)

    def _load_documents(self, filepath):
        _, extension = os.path.splitext(filepath)
        if extension.lower() == '.pdf':
            loader = PyPDFLoader(filepath)
        else:
            # Default to TextLoader for .txt and other file types
            loader = TextLoader(filepath, encoding='utf-8')
        return loader.load()

    def add_document(self, filepath):
        print(f"Processing document: {filepath}")
        try:
            new_docs = self._load_documents(filepath)
            chunks = self.text_splitter.split_documents(new_docs)
            if chunks:
                self.vector_store.add_documents(chunks)
                print(f"Added {len(chunks)} chunks from {filepath} to the knowledge base.")
        except Exception as e:
            print(f"Error processing document {filepath}: {e}")

    def add_new_pathway_documents(self, pathway_data):
        print(f"Processing {len(pathway_data)} documents from Pathway live stream...")
        docs_to_add = []
        for item in pathway_data:
            doc_content = item.get('doc', '')
            # Ensure metadata is a dictionary with a 'source' key
            metadata_path = item.get('metadata', 'live_stream')
            metadata = {'source': os.path.basename(metadata_path)}
            docs_to_add.append(Document(page_content=doc_content, metadata=metadata))
        
        if docs_to_add:
            chunks = self.text_splitter.split_documents(docs_to_add)
            if chunks:
                self.vector_store.add_documents(chunks)
                print(f"Added {len(chunks)} chunks from Pathway to the knowledge base.")

    def analyze_draft(self, draft_content):
        print("Analyzing draft content...")
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(draft_content)

            if not retrieved_docs:
                return {"success": True, "analysis": {"potential_citations": [], "unsupported_claims": [], "related_papers": [], "validation_feedback": []}}

            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            system_message = """
            You are an expert academic research assistant. Analyze an author's draft based *only* on the provided context of source documents.
            Your entire response must be a single, valid JSON object. Do not add any text before or after the JSON.
            If the context is insufficient, return an empty list `[]` for the corresponding key.
            """

            human_template = """
            **Context Documents:**
            {context}
            ---
            **Author's Draft Text:**
            {draft}
            ---
            **Your Task:**
            Analyze the "Author's Draft Text" using *only* the "Context Documents". Generate a JSON object with the following keys:

            1.  "potential_citations": Array of objects. For each claim in the draft directly supported by the context, create an object with:
                - "claim_in_draft": The exact sentence from the author's draft.
                - "supporting_quote_from_context": The specific quote from the context that supports the claim.
                - "source": The source document of the quote.

            2.  "unsupported_claims": Array of strings. Identify factual claims in the draft that *cannot* be verified by the context.

            3.  "related_papers": Array of strings. Suggest up to 3 source documents from the context relevant to the draft's main topic.

            4.  "validation_feedback": Array of objects. Check for inconsistencies between the draft and the context. Create an object with:
                - "draft_statement": The statement from the draft being checked.
                - "feedback": Your analysis (e.g., "This contradicts the context, which states '...'").
                - "source": The relevant source from the context.
            """

            prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_template)])
            chain = prompt | self.llm | StrOutputParser()
            json_string_output = chain.invoke({"context": context, "draft": draft_content})

            try:
                analysis_result = json.loads(json_string_output)
                # Ensure all keys exist in the final output
                analysis_result.setdefault('potential_citations', [])
                analysis_result.setdefault('unsupported_claims', [])
                analysis_result.setdefault('related_papers', [])
                analysis_result.setdefault('validation_feedback', [])
                return {"success": True, "analysis": analysis_result}
            except json.JSONDecodeError:
                print("Error: LLM did not return valid JSON.")
                return {"error": "Failed to parse analysis from AI.", "raw_output": json_string_output}

        except Exception as e:
            print(f"Error during draft analysis: {e}")
            return {"error": "An internal error occurred during analysis."}

    def generate_report(self, question, draft_context=""):
        print(f"Generating report for question: '{question}'")
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(question + "\n" + draft_context)

            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

            if not retrieved_docs:
                return {"summary": "Could not find relevant information to answer the question.", "sources": []}

            prompt_template = """
            Answer the user's question based *only* on the provided "Context Documents" and "User's Draft".
            If the information is not in the context, state that clearly.
            
            Context Documents:
            {context}
            ---
            User's Draft Paper:
            {draft_context}
            ---
            Question: {question}
            Answer:
            """
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"context": context, "draft_context": draft_context, "question": question})
            
            return {"success": True, "summary": summary, "sources": sources}
        except Exception as e:
            print(f"Error during report generation: {e}")
            return {"error": "Failed to generate report."}

    def search_concept(self, topic):
        print(f"Searching for concept: {topic}")
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            results = retriever.invoke(topic)
            # Use a set to avoid duplicate file paths
            unique_papers = {doc.metadata.get('source', 'Unknown') for doc in results}
            return [{"path": paper} for paper in unique_papers if paper != 'Unknown']
        except Exception as e:
            print(f"Error during concept search: {e}")
            return []

    def summarize_document(self, filepath):
        print(f"Summarizing document: {filepath}")
        try:
            if not os.path.exists(filepath):
                return "Error: File not found."
            
            docs = self._load_documents(filepath)
            full_text = " ".join([doc.page_content for doc in docs])
            
            # Take the first ~4000 characters for a concise summary
            text_to_summarize = full_text[:4000]

            prompt_template = "Provide a concise, one-paragraph summary of the following academic text:\n\n{document_text}"
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Use a separate LLM instance without JSON mode for plain text summary
            summary_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
            chain = prompt | summary_llm | StrOutputParser()
            
            summary = chain.invoke({"document_text": text_to_summarize})
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Error: Could not generate summary for the document."


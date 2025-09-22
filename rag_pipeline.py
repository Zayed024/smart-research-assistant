# rag_pipeline.py
import os
import getpass
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from dotenv import load_dotenv

from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

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

# Pydantic models to define the desired JSON structure for the LLM
class Citation(BaseModel):
    claim_in_draft: str = Field(description="The exact sentence from the author's draft.")
    supporting_quote_from_context: str = Field(description="The specific quote from the context that supports the claim.")
    source: str = Field(description="The source document of the quote.")

class Feedback(BaseModel):
    draft_statement: str = Field(description="The statement from the draft being checked.")
    feedback: str = Field(description="Your analysis (e.g., 'This statement appears to contradict the context, which states \'...\'').")
    source: str = Field(description="The relevant source from the context.")

class Analysis(BaseModel):
    potential_citations: List[Citation]
    unsupported_claims: List[str]
    related_papers: List[str]
    validation_feedback: List[Feedback]

class RagPipeline:
    def __init__(self, llm=None):
        if llm is None:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None # Start with no vector store

    def _load_documents(self, filepath):
        _, extension = os.path.splitext(filepath)
        extension = extension.lower()
        if extension == '.pdf':
            loader = PyPDFLoader(filepath)
        elif extension == '.docx':
            loader = Docx2txtLoader(filepath)
        elif extension in ['.png', '.jpg', '.jpeg']:
            return [] # Return empty list, will be handled by OCR
        else:
            loader = TextLoader(filepath, encoding='utf-8')
        return loader.load()

    def _ocr_process_file(self, filepath):
        print(f"File '{filepath}' requires OCR. Processing...")
        _, extension = os.path.splitext(filepath)
        extension = extension.lower()
        text = ""
        try:
            if extension == '.pdf':
                images = convert_from_path(filepath)
                for i, image in enumerate(images):
                    text += f"\n\n--- Page {i+1} ---\n" + pytesseract.image_to_string(image)
            elif extension in ['.png', '.jpg', '.jpeg']:
                text = pytesseract.image_to_string(Image.open(filepath))

            # Create a LangChain Document from the OCR text
            if text.strip():
                return [Document(page_content=text, metadata={'source': filepath})]
        except Exception as e:
            print(f"OCR failed for {filepath}: {e}")
        return []

    def add_document(self, filepath):
        print(f"Processing document: {filepath}")
        try:
            new_docs = self._load_documents(filepath)
            print(f"Regular loader found {len(new_docs)} documents")

            # Check if any documents have actual content
            has_content = any(len(doc.page_content.strip()) > 0 for doc in new_docs)
            print(f"Documents have content: {has_content}")

            # If no documents have content, try OCR
            if not has_content:
                print(f"Regular loader failed to extract content, attempting OCR...")
                new_docs = self._ocr_process_file(filepath)
                print(f"OCR found {len(new_docs)} documents")

            chunks = self.text_splitter.split_documents(new_docs)
            print(f"Number of chunks created: {len(chunks)}")

            if not chunks:
                print(f"Warning: No text chunks extracted from {filepath}, even after OCR attempt.")
                return False

            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                print(f"Created new vector store with {len(chunks)} chunks from {filepath}.")
            else:
                self.vector_store.add_documents(chunks)
                print(f"Added {len(chunks)} chunks from {filepath} to the knowledge base.")

            return True
        except Exception as e:
            print(f"Error processing document {filepath}: {e}")
            return False

    def add_new_pathway_documents(self, pathway_data):
        print(f"Processing {len(pathway_data)} documents from Pathway live stream...")
        docs_to_add = []
        for item in pathway_data:
            doc_content = item.get('doc', '')
            metadata = {'source': item.get('metadata', 'live_stream')}
            docs_to_add.append(Document(page_content=doc_content, metadata=metadata))
        if docs_to_add:
            chunks = self.text_splitter.split_documents(docs_to_add)
            if chunks:
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                    print(f"Created new vector store with {len(chunks)} chunks from Pathway.")
                else:
                    self.vector_store.add_documents(chunks)
                    print(f"Added {len(chunks)} chunks from Pathway to the knowledge base.")

    def _extract_json_from_response(self, response_text):
        try:
            cleaned = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                return json.loads(cleaned[start_idx:end_idx])
            else:
                return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match: return json.loads(json_match.group())
            except: pass
            raise json.JSONDecodeError("Could not extract valid JSON from response", response_text, 0)

    

    def analyze_draft(self, draft_content):
        print(f"Analyzing draft content...")
        try:
            if self.vector_store is None:
                return {"success": False, "error": "Please upload a source document first."}

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(draft_content)

            retrieved_context_for_ui = [
                {"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content}
                for doc in retrieved_docs
            ]

            if not retrieved_docs:
                return {"success": True, "analysis": {}, "retrieved_context": []}

            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

            
            # 1. Bind the Pydantic model to the LLM to enforce JSON mode
            structured_llm = self.llm.with_structured_output(Analysis)

            # 2. The prompt is now simpler, as the schema is defined in the model
            system_message = "You are an expert academic research assistant..." # Your original prompt is fine here
            human_template = """
            Analyze the "Author's Draft Text" using *only* the "Context Documents". Generate a structured analysis.
            Context Documents: {context}
            ---
            Author's Draft Text: {draft}
            """

            prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_template)])

            # 3. The chain now uses the structured LLM and doesn't need a string parser
            chain = prompt | structured_llm

            # 4. The result is now a clean Pydantic object, not a string
            analysis_result_object = chain.invoke({"context": context, "draft": draft_content})

            # Convert Pydantic object to a dictionary for the API response
            analysis_result = analysis_result_object.dict()

            return {"success": True, "analysis": analysis_result, "retrieved_context": retrieved_context_for_ui}

        except Exception as e:
            import traceback
            print(f"Error during draft analysis: {e}")
            traceback.print_exc()
            return {"error": "An internal error occurred during analysis."}

    def generate_report(self, question, draft_context=""):
        print(f"Generating report for question: '{question}'")
        try:
            if self.vector_store is None:
                return {"success": False, "error": "Please upload a source document first."}

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(question)

            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

            if not retrieved_docs:
                return {"success": True, "summary": "Could not find relevant information to answer the question.", "sources": [], "attribution": []}

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

            # Get attribution information for the sources
            attribution = self._get_document_attribution(sources)

            return {"success": True, "summary": summary, "sources": sources, "attribution": attribution}
        except Exception as e:
            print(f"Error during report generation: {e}")
            return {"error": "Failed to generate report."}

    def _get_document_attribution(self, sources):
        """Get attribution information for documents from the database"""
        if not sources:
            return []

        attribution_info = []
        try:
            import sqlite3
            conn = sqlite3.connect('research_assistant.db')
            conn.row_factory = sqlite3.Row

            # Create a query to get attribution for multiple sources
            placeholders = ','.join(['?'] * len(sources))
            query = f'''
                SELECT
                    d.id, d.filename, d.title, d.uploaded_at,
                    u.username as uploaded_by_username,
                    d.description
                FROM documents d
                JOIN users u ON d.uploaded_by = u.id
                WHERE d.file_path IN ({placeholders})
                ORDER BY d.uploaded_at DESC
            '''

            cursor = conn.execute(query, sources)
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                attribution_info.append({
                    "document_id": row['id'],
                    "filename": row['filename'],
                    "title": row['title'] or row['filename'],
                    "uploaded_by": row['uploaded_by_username'],
                    "uploaded_at": row['uploaded_at'],
                    "description": row['description'] or ""
                })

        except Exception as e:
            print(f"Error getting document attribution: {e}")
            # Fallback to basic source information
            for source in sources:
                attribution_info.append({
                    "filename": source,
                    "title": source,
                    "uploaded_by": "Unknown",
                    "uploaded_at": "Unknown",
                    "description": ""
                })

        return attribution_info

    def search_concept(self, topic):
        print(f"Searching for concept: {topic}")
        try:
            if self.vector_store is None:
                return []
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

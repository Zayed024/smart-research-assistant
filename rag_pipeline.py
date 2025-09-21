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

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
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
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None # Start with no vector store
        self.llm = llm

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
            combined_query = question
            if draft_context:
                combined_query += "\n\nRelevant context from the user's current work:\n" + draft_context
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(combined_query)
            if not retrieved_docs:
                return {"success": True, "summary": "Could not find relevant information to answer.", "sources": []}
            source_objects = [
                {"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content}
                for doc in retrieved_docs
            ]
            unique_sources = list({v['content']:v for v in source_objects}.values())
            context = "\n\n---\n\n".join([doc["content"] for doc in unique_sources])
            system_message = "You are a smart research assistant. Answer the user's question concisely based *only* on the provided context. If the answer isn't in the context, say so."
            human_message = "Context Documents:\n{context}\n---\nQuestion: {question}"
            prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])
            output_parser = StrOutputParser()
            chain = prompt | self.llm | output_parser
            summary = chain.invoke({"context": context, "question": question})
            return {"success": True, "summary": summary, "sources": unique_sources}
        except Exception as e:
            print(f"Error during report generation: {e}")
            return {"error": "Failed to generate report."}

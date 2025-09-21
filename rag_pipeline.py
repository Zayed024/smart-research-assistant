# # rag_pipeline.py
# import os
# import getpass
# import json
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document
# from dotenv import load_dotenv

# load_dotenv()

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.1,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# class RagPipeline:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         self.vector_store = None
#         self._initialize_vector_store()
#         self.llm = llm
#     def _initialize_vector_store(self):
#         placeholder_text = ["Initialize with some text to create the vector store"]
#         self.vector_store = FAISS.from_texts(placeholder_text, self.embeddings)

#     def _load_documents(self, filepath):
#         _, extension = os.path.splitext(filepath)
#         if extension.lower() == '.pdf':
#             loader = PyPDFLoader(filepath)
#         else:
#             loader = TextLoader(filepath, encoding='utf-8')
#         return loader.load()

#     def add_document(self, filepath):
#         print(f"Processing document: {filepath}")
#         try:
#             new_docs = self._load_documents(filepath)
#             chunks = self.text_splitter.split_documents(new_docs)
#             if chunks:
#                 self.vector_store.add_documents(chunks)
#                 print(f"Added {len(chunks)} chunks from {filepath} to the knowledge base.")
#         except Exception as e:
#             print(f"Error processing document {filepath}: {e}")

#     def add_new_pathway_documents(self, pathway_data):
#         print(f"Processing {len(pathway_data)} documents from Pathway live stream...")
#         docs_to_add = []
#         for item in pathway_data:
#             doc_content = item.get('doc', '')
#             metadata = {'source': item.get('metadata', 'live_stream')}
#             docs_to_add.append(Document(page_content=doc_content, metadata=metadata))

#         if docs_to_add:
#             chunks = self.text_splitter.split_documents(docs_to_add)
#             if chunks:
#                 self.vector_store.add_documents(chunks)
#                 print(f"Added {len(chunks)} chunks from Pathway to the knowledge base.")

#     def analyze_draft(self, draft_content):
#         print(f"Analyzing draft content...")
#         try:
#             retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
#             retrieved_docs = retriever.invoke(draft_content)

#             if not retrieved_docs:
#                 return {"success": True, "analysis": {"potential_citations": [], "unsupported_claims": [], "related_papers": [], "validation_feedback": []}}

#             context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

#             system_message = """
#             You are an expert academic research assistant. Your task is to analyze an author's draft text based *only* on a provided context of source documents. Your goal is to help them improve their paper by suggesting citations, identifying related work, and validating their claims against the sources.

#             Follow these rules strictly:
#             1.  Base all your output *exclusively* on the provided 'Context Documents'. Do not use outside knowledge.
#             2.  Your entire response must be a single, valid JSON object. Do not add any text before or after the JSON.
#             3.  If the context is insufficient to perform a task, return an empty list `[]` for that key.
#             """

#             human_template = """
#             Here is the information for your analysis:

#             **Context Documents:**
#             {context}

#             ---

#             **Author's Draft Text:**
#             {draft}

#             ---

#             **Your Task:**
#             Analyze the "Author's Draft Text" using *only* the "Context Documents". Generate a JSON object with the following keys:

#             1.  "potential_citations": An array of objects. For each claim in the draft that is directly supported by the context, create an object with:
#                 - "claim_in_draft": The exact sentence from the author's draft.
#                 - "supporting_quote_from_context": The specific quote from the context that supports the claim.
#                 - "source": The source document of the quote (e.g., the filename).

#             2.  "unsupported_claims": An array of strings. Identify factual claims or statements in the draft that *cannot* be verified or supported by the provided "Context Documents". List the exact sentences from the draft. This helps the author identify where they might need additional citations.

#             3.  "related_papers": An array of strings. Suggest up to 3 source documents from the context that are highly relevant to the draft's main topic. List their source names.

#             4.  "validation_feedback": An array of objects. Check for inconsistencies or contradictions between the draft and the context. For each finding, create an object with:
#                 - "draft_statement": The statement from the draft being checked.
#                 - "feedback": Your analysis (e.g., "This statement appears to contradict the context, which states '...'").
#                 - "source": The relevant source from the context.
#                 If there are no inconsistencies, return an empty array.

#             JSON Output:
#             """

#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", system_message),
#                 ("human", human_template),
#             ])

#             chain = prompt | self.llm | StrOutputParser()

#             json_string_output = chain.invoke({
#                 "context": context,
#                 "draft": draft_content
#             })

#             try:
#                 analysis_result = json.loads(json_string_output)
#                 analysis_result.setdefault('potential_citations', [])
#                 analysis_result.setdefault('unsupported_claims', [])
#                 analysis_result.setdefault('related_papers', [])
#                 analysis_result.setdefault('validation_feedback', [])
#                 return {"success": True, "analysis": analysis_result}
#             except json.JSONDecodeError:
#                 print("Error: LLM did not return valid JSON.")
#                 return {"error": "Failed to parse analysis from AI.", "raw_output": json_string_output}

#         except Exception as e:
#             print(f"Error during draft analysis: {e}")
#             return {"error": "An internal error occurred during analysis."}

#     def generate_report(self, question, draft_context=""):
#         print(f"Generating report for question: '{question}'")
#         try:
#             combined_query = question
#             if draft_context:
#                 combined_query += "\n\nRelevant context from the user's current work:\n" + draft_context

#             retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
#             retrieved_docs = retriever.invoke(combined_query)

#             context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
#             sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

#             if not retrieved_docs:
#                  return {"summary": "Could not find relevant information in the uploaded documents to answer the question.", "sources": []}

#             system_message = """
#             You are a smart research assistant. Your task is to answer a user's question based on two sources of information:
#             1. A provided context of source documents.
#             2. The user's own draft paper text.
#             Answer concisely and base your answer *only* on the provided information. If the answer isn't in the context, say so.
#             """

#             human_message = """
#             Context Documents:
#             {context}
#             ---
#             User's Draft Paper:
#             {draft_context}
#             ---
#             Question: {question}
#             """

#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", system_message),
#                 ("human", human_message)
#             ])

#             output_parser = StrOutputParser()
#             chain = prompt | self.llm | output_parser

#             summary = chain.invoke({
#                 "context": context,
#                 "draft_context": draft_context,
#                 "question": question
#             })

#             return {"success": True, "summary": summary, "sources": sources}
#         except Exception as e:
#             print(f"Error during report generation: {e}")
#             return {"error": "Failed to generate report."}


# rag_pipeline.py
import os
import getpass
import json
import re
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

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class RagPipeline:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_vector_store()
        self.llm = llm

    def _initialize_vector_store(self):
        placeholder_text = ["Initialize with some text to create the vector store"]
        self.vector_store = FAISS.from_texts(placeholder_text, self.embeddings)

    def _load_documents(self, filepath):
        _, extension = os.path.splitext(filepath)
        if extension.lower() == '.pdf':
            loader = PyPDFLoader(filepath)
        else:
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
            metadata = {'source': item.get('metadata', 'live_stream')}
            docs_to_add.append(Document(page_content=doc_content, metadata=metadata))

        if docs_to_add:
            chunks = self.text_splitter.split_documents(docs_to_add)
            if chunks:
                self.vector_store.add_documents(chunks)
                print(f"Added {len(chunks)} chunks from Pathway to the knowledge base.")

    def _extract_json_from_response(self, response_text):
        """Extract JSON from LLM response, handling various formatting issues"""
        try:
            # Clean the response text
            cleaned = response_text.strip()

            # Remove markdown code blocks if present
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]

            cleaned = cleaned.strip()

            # Try to find JSON object boundaries
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_part = cleaned[start_idx:end_idx]
                return json.loads(json_part)
            else:
                return json.loads(cleaned)

        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using regex
            try:
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

            raise json.JSONDecodeError("Could not extract valid JSON from response", cleaned, 0)

    def analyze_draft(self, draft_content):
        print(f"Analyzing draft content...")
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(draft_content)

            if not retrieved_docs:
                return {"success": True, "analysis": {"potential_citations": [], "unsupported_claims": [], "related_papers": [], "validation_feedback": []}}

            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

            system_message = """
            You are an expert academic research assistant. Your task is to analyze an author's draft text based *only* on a provided context of source documents. Your goal is to help them improve their paper by suggesting citations, identifying related work, and validating their claims against the sources.

            Follow these rules strictly:
            1.  Base all your output *exclusively* on the provided 'Context Documents'. Do not use outside knowledge.
            2.  Your entire response must be a single, valid JSON object. Do not add any text before or after the JSON.
            3.  If the context is insufficient to perform a task, return an empty list `[]` for that key.
            """

            human_template = """
            Here is the information for your analysis:

            **Context Documents:**
            {context}

            ---

            **Author's Draft Text:**
            {draft}

            ---

            **Your Task:**
            Analyze the "Author's Draft Text" using *only* the "Context Documents". Generate a JSON object with the following keys:

            1.  "potential_citations": An array of objects. For each claim in the draft that is directly supported by the context, create an object with:
                - "claim_in_draft": The exact sentence from the author's draft.
                - "supporting_quote_from_context": The specific quote from the context that supports the claim.
                - "source": The source document of the quote (e.g., the filename).

            2.  "unsupported_claims": An array of strings. Identify factual claims or statements in the draft that *cannot* be verified or supported by the provided "Context Documents". List the exact sentences from the draft. This helps the author identify where they might need additional citations.

            3.  "related_papers": An array of strings. Suggest up to 3 source documents from the context that are highly relevant to the draft's main topic. List their source names.

            4.  "validation_feedback": An array of objects. Check for inconsistencies or contradictions between the draft and the context. For each finding, create an object with:
                - "draft_statement": The statement from the draft being checked.
                - "feedback": Your analysis (e.g., "This statement appears to contradict the context, which states '...'").
                - "source": The relevant source from the context.
                If there are no inconsistencies, return an empty array.

            JSON Output:
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", human_template),
            ])

            chain = prompt | self.llm | StrOutputParser()

            json_string_output = chain.invoke({
                "context": context,
                "draft": draft_content
            })

            try:
                analysis_result = self._extract_json_from_response(json_string_output)
                analysis_result.setdefault('potential_citations', [])
                analysis_result.setdefault('unsupported_claims', [])
                analysis_result.setdefault('related_papers', [])
                analysis_result.setdefault('validation_feedback', [])
                return {"success": True, "analysis": analysis_result}
            except json.JSONDecodeError as e:
                print(f"Error: LLM did not return valid JSON. Raw output: {json_string_output[:500]}...")
                print(f"JSON parsing error: {e}")
                # Return a fallback response instead of error
                return {
                    "success": True,
                    "analysis": {
                        "potential_citations": [],
                        "unsupported_claims": [],
                        "related_papers": [],
                        "validation_feedback": []
                    },
                    "note": "Analysis could not be completed due to formatting issues"
                }

        except Exception as e:
            print(f"Error during draft analysis: {e}")
            return {"error": "An internal error occurred during analysis."}

    def generate_report(self, question, draft_context=""):
        print(f"Generating report for question: '{question}'")
        try:
            combined_query = question
            if draft_context:
                combined_query += "\n\nRelevant context from the user's current work:\n" + draft_context

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(combined_query)

            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

            if not retrieved_docs:
                 return {"summary": "Could not find relevant information in the uploaded documents to answer the question.", "sources": []}

            system_message = """
            You are a smart research assistant. Your task is to answer a user's question based on two sources of information:
            1. A provided context of source documents.
            2. The user's own draft paper text.
            Answer concisely and base your answer *only* on the provided information. If the answer isn't in the context, say so.
            """

            human_message = """
            Context Documents:
            {context}
            ---
            User's Draft Paper:
            {draft_context}
            ---
            Question: {question}
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", human_message)
            ])

            output_parser = StrOutputParser()
            chain = prompt | self.llm | output_parser

            summary = chain.invoke({
                "context": context,
                "draft_context": draft_context,
                "question": question
            })

            return {"success": True, "summary": summary, "sources": sources}
        except Exception as e:
            print(f"Error during report generation: {e}")
            return {"error": "Failed to generate report."}

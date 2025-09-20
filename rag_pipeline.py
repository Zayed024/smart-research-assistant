# rag_pipeline.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import getpass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
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


         # --- Memory store (session_id -> ChatMessageHistory) ---
        self._session_store: dict[str, BaseChatMessageHistory] = {}

        # --- Parsers and prompt parts reused across calls ---
        self.json_parser = JsonOutputParser()

        # We include a MessagesPlaceholder so chat history is injected automatically.
        self.system_message = """
You are a smart research assistant. Your task is to answer a user's question based ONLY on the provided context.
Analyze the context and generate a report in a JSON format. The JSON object must contain two keys:
1. "summary": A concise, well-structured summary of the answer.
2. "takeaways": A JSON array of 3 to 4 key bullet points or takeaways from the summary.

Follow these rules strictly:
- Base your entire response on the provided context. Do not use outside information.
- If the context does not contain the answer, return a JSON object with an empty summary and an empty takeaways array.
- Your final output MUST be a valid JSON object and nothing else.

Format instructions: {format_instructions}
""".strip()

        # The prompt includes prior turns via {history}, then current context + question.
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context:\n{context}\n---\nQuestion: {question}"),
        ]).partial(format_instructions=self.json_parser.get_format_instructions())


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

    def _get_session_history(self, config) -> BaseChatMessageHistory:
        # Extract session_id from config, with fallback logic
        if isinstance(config, dict):
            session_id = (config.get("configurable") or {}).get("session_id", "default")
        else:
            # Handle case where config is not a dict (e.g., passed as string)
            session_id = "default"

        if session_id not in self._session_store:
            self._session_store[session_id] = ChatMessageHistory()
        return self._session_store[session_id]


    def generate_report(self, question: str, session_id: str = "default"):
        """
        Generates a report using RAG with conversational memory.
        - session_id controls which chat history to use/append to.
        """

        print(f"Generating report for question: '{question}'")
        try:
            # 1. Retrieve relevant documents
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(question)
            if not retrieved_docs:
                context = ""
                sources = []
            else:
                context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

            # 2) Build per-call chain, then wrap with history
            base_chain = self.prompt_template | self.llm | self.json_parser

            chain_with_history = RunnableWithMessageHistory(
                base_chain,
                get_session_history=self._get_session_history,
                input_messages_key="question",
                history_messages_key="history"
            )

            # 3) Invoke with memory-aware config - FIXED: Pass config as keyword argument
            report_json = chain_with_history.invoke(
                {"context": context, "question": question},
                config={"configurable": {"session_id": session_id}}
            )

            # --- START OF DEBUG BLOCK ---
            print("--- DEBUG INFO ---")
            print(f"DEBUG: Type of report_json is {type(report_json)}")
            print(f"DEBUG: Value is {report_json}")
            print("--- END OF DEBUG BLOCK ---")
            # --- END OF DEBUG BLOCK ---

            # 4) Handle the response - add robust error handling for JSON parsing
            if isinstance(report_json, dict):
                # Normal case: report_json is already a dictionary
                return {
                    "success": True,
                    "summary": report_json.get("summary", "No summary generated."),
                    "takeaways": report_json.get("takeaways", []),
                    "sources": sources,
                }
            elif isinstance(report_json, str):
                # Handle case where JsonOutputParser returns a string instead of dict
                print(f"WARNING: JsonOutputParser returned a string instead of dict. Attempting to parse as JSON.")
                try:
                    import json
                    parsed_json = json.loads(report_json)
                    if isinstance(parsed_json, dict):
                        return {
                            "success": True,
                            "summary": parsed_json.get("summary", "No summary generated."),
                            "takeaways": parsed_json.get("takeaways", []),
                            "sources": sources,
                        }
                    else:
                        print(f"ERROR: Parsed JSON is not a dictionary: {type(parsed_json)}")
                        return {
                            "success": False,
                            "error": "Invalid response format from AI model.",
                            "summary": "Unable to generate summary due to response format issue.",
                            "takeaways": [],
                            "sources": sources,
                        }
                except json.JSONDecodeError as json_error:
                    print(f"ERROR: Failed to parse JSON string: {json_error}")
                    print(f"Raw string content: {report_json[:500]}...")  # Log first 500 chars
                    return {
                        "success": False,
                        "error": "Failed to parse AI model response as JSON.",
                        "summary": "Unable to generate summary due to parsing error.",
                        "takeaways": [],
                        "sources": sources,
                    }
            else:
                # Handle unexpected type
                print(f"ERROR: Unexpected type returned from JsonOutputParser: {type(report_json)}")
                return {
                    "success": False,
                    "error": f"Unexpected response type: {type(report_json)}",
                    "summary": "Unable to generate summary due to unexpected response type.",
                    "takeaways": [],
                    "sources": sources,
                }

        except Exception as e:
            print(f"Error during report generation: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Failed to generate report: {str(e)}",
                "summary": "Unable to generate summary due to an error.",
                "takeaways": [],
                "sources": [],
            }

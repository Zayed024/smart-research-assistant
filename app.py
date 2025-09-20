import os
import sqlite3
import threading
import time
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pathway as pw
from rag_pipeline import RagPipeline 
from PyPDF2 import PdfReader 
from pathway.io.python import ConnectorObserver
from datetime import datetime

class PathwayObserver(ConnectorObserver):
    def __init__(self, holder, batch_size=5, flush_interval=3):
        self.holder = holder
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush_time = datetime.now()

    def on_change(self, key, row, time, is_addition):
        if is_addition:
            self.buffer.append(row)
            if len(self.buffer) >= self.batch_size:
                self.flush()

    def on_time_end(self, time):
        # Flush if enough time has passed since last flush
        if (datetime.now() - self.last_flush_time).total_seconds() >= self.flush_interval:
            self.flush()

    def flush(self):
        if self.buffer:
            print(f"Processing {len(self.buffer)} documents from Pathway live stream...")
            self.holder.update_data(self.buffer)
            self.buffer.clear()
            self.last_flush_time = pw.now()

    def on_end(self):
        self.flush()
        print("Stream ended.")

def decode_file(data_bytes, file_path:str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(data_bytes))
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            return text
        except Exception as e:
            print(f"[PDF ERROR] Could not parse {file_path}: {e}")
            return ""
    else:
        # Assume text file
        return data_bytes.decode("utf-8", errors="ignore")

# --- Configuration ---
UPLOAD_FOLDER = 'uploaded_files'
LIVE_DATA_FOLDER = 'live_data_source'
DATABASE = 'research_assistant.db'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LIVE_DATA_FOLDER'] = LIVE_DATA_FOLDER

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIVE_DATA_FOLDER, exist_ok=True)

# --- Database Setup ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with open('schema.sql', 'r') as f:
        conn.executescript(f.read())
    # Initialize counters if they don't exist
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO usage_stats (id, questions_asked, reports_generated) VALUES (1, 0, 0)")
    conn.commit()
    conn.close()

# --- Pathway Live Data Ingestion ---
# This class will hold the results from our Pathway pipeline
class PathwayResultHolder:
    def __init__(self):
        self.data = []
        self.lock = threading.Lock()

    def update_data(self, new_data):
        with self.lock:
            # This is a simplistic update; in a real app, you'd merge intelligently
            self.data = new_data
            print(f"Pathway data updated: {len(self.data)} items")
            # Update the main RAG pipeline's knowledge base with this new data
            rag_pipeline.add_new_pathway_documents(self.data)


pathway_results = PathwayResultHolder()

def run_pathway_pipeline():
    """
    Defines and runs the Pathway pipeline to process live data.
    """
    class InputSchema(pw.Schema):
        doc: str
        metadata: str

    def process_live_data(table):
        # In a real scenario, you would embed and process the data here
        # For this example, we are just passing it through
        # Decode based on file type
        return table.select(
            doc=pw.apply(lambda b, p: decode_file(b, str(p)), pw.this.data, pw.this._metadata["path"]),
            metadata=pw.this._metadata["path"]
        )

    rd = pw.io.fs.read(
        app.config['LIVE_DATA_FOLDER'],
        format="binary",
        mode="streaming",
        with_metadata=True,
        name="live_data_input"
    )
    #print(rd.schema)

    decoded_rd = process_live_data(rd)
    #decoded_rd = rd.select(
        # doc=pw.apply(lambda b: b.decode("utf-8", errors="ignore"), pw.this.data),
        # metadata=pw.this._metadata["path"]  
    #)
    
    
    observer = PathwayObserver(pathway_results, batch_size=5, flush_interval=3)
    pw.io.python.write(decoded_rd, observer)

    print("Starting Pathway pipeline...")
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)


# --- RAG Pipeline Initialization ---
# Initialize the main RAG pipeline
rag_pipeline = RagPipeline()

# --- Flask API Routes ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process and add the file to our RAG knowledge base
        rag_pipeline.add_document(filepath)
        
        return jsonify({"success": f"File '{filename}' uploaded and processed."}), 200
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question not provided"}), 400
    
    question = data['question']
    
    # Use the RAG pipeline to get an answer
    report = rag_pipeline.generate_report(question)
    
    # Update usage stats
    conn = get_db_connection()
    conn.execute("UPDATE usage_stats SET questions_asked = questions_asked + 1 WHERE id = 1")
    if report.get("success"):
         conn.execute("UPDATE usage_stats SET reports_generated = reports_generated + 1 WHERE id = 1")
    conn.commit()
    conn.close()
    
    return jsonify(report)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = get_db_connection()
    stats = conn.execute("SELECT * FROM usage_stats WHERE id = 1").fetchone()
    conn.close()
    if stats:
        return jsonify(dict(stats))
    return jsonify({"questions_asked": 0, "reports_generated": 0})

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize the database
    init_db()

    # Start the Pathway pipeline in a separate thread
    pathway_thread = threading.Thread(target=run_pathway_pipeline, daemon=True)
    pathway_thread.start()
    
    # Add a dummy file to the live folder to show the pipeline is working
    with open(os.path.join(LIVE_DATA_FOLDER, 'initial_news.txt'), 'w') as f:
        f.write("This is a sample live news update from Pathway. The stock market is showing volatile trends today.")

    # Start the Flask app
    app.run(debug=True, use_reloader=False) # use_reloader=False is important for threading

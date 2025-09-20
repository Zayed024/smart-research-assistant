import os
import sqlite3
import threading
from flask import Flask, request, jsonify, render_template,session
from werkzeug.utils import secure_filename
import pathway as pw
from rag_pipeline import RagPipeline
import uuid
from pathway.io.python import ConnectorObserver

# --- Configuration (No changes here) ---
UPLOAD_FOLDER = 'uploaded_files'
LIVE_DATA_FOLDER = 'live_data_source'
DATABASE = 'research_assistant.db'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LIVE_DATA_FOLDER'] = LIVE_DATA_FOLDER
app.secret_key = os.urandom(24)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIVE_DATA_FOLDER, exist_ok=True)

# --- Database Setup (No changes here) ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with open('schema.sql', 'r') as f:
        conn.executescript(f.read())
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO usage_stats (id, questions_asked, reports_generated) VALUES (1, 0, 0)")
    conn.commit()
    conn.close()



class SimplePathwayObserver(ConnectorObserver):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def on_change(self, key, row, time, is_addition):
        # This method is called by Pathway for each new document.
        if is_addition:
            print(f"Received document from Pathway: {row['metadata']}")
            # We pass the data as a list containing the single new row (document).
            self.pipeline.add_new_pathway_documents([row])


def run_pathway_pipeline():
    
    rd = pw.io.fs.read(
        app.config['LIVE_DATA_FOLDER'],
        format="binary",
        mode="streaming",
        with_metadata=True,
    )

    decoded_rd = rd.select(
        doc=pw.apply(lambda b: b.decode("utf-8", errors="ignore"), pw.this.data),
        metadata=pw.this._metadata["path"]
    )

    
    observer = SimplePathwayObserver(rag_pipeline)

    # Pass the observer OBJECT to the write function
    pw.io.python.write(decoded_rd, observer)

    print("Starting Pathway pipeline...")
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)



rag_pipeline = RagPipeline()

# --- Flask API Routes ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    session.clear()
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
      # Get or create a session ID for the conversation
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']

    report = rag_pipeline.generate_report(question, session_id=session_id)

    report['session_id'] = session_id
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

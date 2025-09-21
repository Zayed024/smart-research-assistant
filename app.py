import os
import sqlite3
import threading
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pathway as pw
from rag_pipeline import RagPipeline, llm 
from PyPDF2 import PdfReader 
from pathway.io.python import ConnectorObserver
from datetime import datetime
from dotenv import load_dotenv
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

load_dotenv()

# --- App & DB Configuration ---
UPLOAD_FOLDER = 'uploaded_files'
LIVE_DATA_FOLDER = 'live_data_source'
DATABASE = 'research_assistant.db'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LIVE_DATA_FOLDER'] = LIVE_DATA_FOLDER
app.config['SECRET_KEY'] = os.urandom(24) # Needed for session management

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIVE_DATA_FOLDER, exist_ok=True)

# --- Authentication Setup ---
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user_row:
        return User(id=user_row['id'], username=user_row['username'])
    return None

# --- Database ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE); conn.row_factory = sqlite3.Row; return conn

def init_db():
    conn = get_db_connection()
    if os.path.exists('schema.sql'):
        with open('schema.sql', 'r') as f: conn.executescript(f.read())
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO usage_stats (id, questions_asked, reports_generated) VALUES (1, 0, 0)")
        conn.commit()
    conn.close()

# --- Pathway & RAG Pipeline ---
# (PathwayObserver, decode_file, PathwayResultHolder, and run_pathway_pipeline are unchanged)
class PathwayObserver(ConnectorObserver):
    def __init__(self, holder, batch_size=5, flush_interval=3):
        self.holder = holder
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush_time = datetime.now()

    def on_change(self, key, row, time, is_addition):
        if is_addition: self.buffer.append(row)
        if len(self.buffer) >= self.batch_size: self.flush()

    def on_time_end(self, time):
        if (datetime.now() - self.last_flush_time).total_seconds() >= self.flush_interval: self.flush()

    def flush(self):
        if self.buffer:
            self.holder.update_data(self.buffer)
            self.buffer.clear()
            self.last_flush_time = pw.now()

    def on_end(self): self.flush()

def decode_file(data_bytes, file_path:str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(data_bytes))
            return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        except Exception: return ""
    else: return data_bytes.decode("utf-8", errors="ignore")

class PathwayResultHolder:
    def __init__(self): self.data = []; self.lock = threading.Lock()
    def update_data(self, new_data):
        with self.lock: self.data = new_data; rag_pipeline.add_new_pathway_documents(self.data)

pathway_results = PathwayResultHolder()
rag_pipeline = RagPipeline(llm=llm)

def run_pathway_pipeline():
    # ... (unchanged)
    pw.io.fs.read(app.config['LIVE_DATA_FOLDER'], format="binary", mode="streaming", with_metadata=True)
    # ...
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)


# --- General API Routes ---
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def index(): return render_template('index.html')
# ... (all previous general routes like /api/config, /api/upload, /api/live_analyze, etc., remain here and unchanged)
@app.route('/api/config')
def get_config():
    api_key = os.getenv('SERPAPI_API_KEY')
    return jsonify({"serpapi_api_key": api_key}) if api_key else (jsonify({"error": "SERPAPI_API_KEY not found"}), 500)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # ... (unchanged)
    return jsonify({"success": "File uploaded."}), 200

@app.route('/api/live_analyze', methods=['POST'])
def live_analyze():
    # ... (unchanged)
    return jsonify(rag_pipeline.analyze_draft(request.get_json().get('draft_content', '')))

@app.route('/api/ask', methods=['POST'])
def ask_question():
    # ... (unchanged)
    return jsonify(rag_pipeline.generate_report(request.get_json()['question'], request.get_json().get('draft_context', '')))

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # ... (unchanged)
    return jsonify({})

# --- NEW: AUTHENTICATION ROUTES ---
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password: return jsonify({"error": "Username and password are required."}), 400
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if user: conn.close(); return jsonify({"error": "Username already exists."}), 409
    
    hashed_password = generate_password_hash(password)
    conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed_password))
    conn.commit()
    conn.close()
    return jsonify({"success": "User registered successfully."}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    conn = get_db_connection()
    user_row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    
    if user_row and check_password_hash(user_row['password_hash'], password):
        user = User(id=user_row['id'], username=user_row['username'])
        login_user(user)
        return jsonify({"success": True, "username": user.username})
    return jsonify({"error": "Invalid username or password."}), 401

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"success": True})

@app.route('/api/auth/status')
def auth_status():
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "username": current_user.username})
    return jsonify({"logged_in": False})

# --- PROTECTED: LEARNING MODULE ENDPOINTS ---
@app.route('/api/learn', methods=['POST'])
@login_required
def learn_concept():
    topic = request.get_json().get('topic')
    if not topic: return jsonify({"error": "Topic not provided"}), 400
    return jsonify({"success": True, "papers": rag_pipeline.search_concept(topic)})

@app.route('/api/summarize_paper', methods=['POST'])
@login_required
def summarize_paper():
    filepath = request.get_json().get('filepath')
    if not filepath: return jsonify({"error": "Filepath not provided"}), 400
    summary = rag_pipeline.summarize_document(filepath)
    if "Error:" in summary: return jsonify({"error": summary})
    return jsonify({"success": True, "summary": summary})

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# --- NEW: PROTECTED LIBRARY ENDPOINTS ---
@app.route('/api/library', methods=['GET'])
@login_required
def get_library():
    conn = get_db_connection()
    papers = conn.execute('SELECT file_path, file_name FROM library WHERE user_id = ?', (current_user.id,)).fetchall()
    conn.close()
    return jsonify([dict(p) for p in papers])

@app.route('/api/library', methods=['POST'])
@login_required
def add_to_library():
    data = request.get_json()
    path = data.get('path')
    name = path.split('/').pop()
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO library (user_id, file_path, file_name) VALUES (?, ?, ?)', (current_user.id, path, name))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Paper already in library."}), 409
    finally:
        conn.close()
    return jsonify({"success": True})

# --- Main Execution ---
if __name__ == '__main__':
    init_db()
    pathway_thread = threading.Thread(target=run_pathway_pipeline, daemon=True)
    pathway_thread.start()
    app.run(debug=True, use_reloader=False)


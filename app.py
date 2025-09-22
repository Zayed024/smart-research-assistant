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

# --- Authentication Setup ---
login_manager = LoginManager()

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

def decode_file(data_bytes, file_path:str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(data_bytes))
            return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        except Exception as e:
            print(f"[PDF ERROR] Could not parse {file_path}: {e}")
            return ""
    else:
        return data_bytes.decode("utf-8", errors="ignore")

# --- Configuration ---
UPLOAD_FOLDER = 'uploaded_files'
LIVE_DATA_FOLDER = 'live_data_source'
DATABASE = 'research_assistant.db'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LIVE_DATA_FOLDER'] = LIVE_DATA_FOLDER
app.config['SECRET_KEY'] = os.urandom(24) # Needed for session management

# Initialize login manager with app
login_manager.init_app(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIVE_DATA_FOLDER, exist_ok=True)

# --- OCR Processing Status Tracking ---
ocr_processing_status = {}

# --- Database ---
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

# --- Pathway Live Data Ingestion ---
class PathwayResultHolder:
    def __init__(self):
        self.data = []
        self.lock = threading.Lock()

    def update_data(self, new_data):
        with self.lock:
            self.data = new_data
            rag_pipeline.add_new_pathway_documents(self.data)

pathway_results = PathwayResultHolder()

def run_pathway_pipeline():
    class InputSchema(pw.Schema):
        doc: str
        metadata: str

    rd = pw.io.fs.read(
        app.config['LIVE_DATA_FOLDER'],
        format="binary",
        mode="streaming",
        with_metadata=True,
    )

    decoded_rd = rd.select(
        doc=pw.apply(lambda b, p: decode_file(b, str(p)), pw.this.data, pw.this._metadata["path"]),
        metadata=pw.this._metadata["path"]
    )

    observer = PathwayObserver(pathway_results, batch_size=2, flush_interval=5)
    pw.io.python.write(decoded_rd, observer)

    print("Starting Pathway pipeline...")
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)

# --- RAG Pipeline Initialization ---
rag_pipeline = RagPipeline(llm=llm)

# --- Flask API Routes ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# Serve top-level static assets (CSS/JS) referenced by index.html
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

@app.route('/styles.css')
def styles():
    return send_from_directory(BASE_DIR, 'styles.css')

@app.route('/script.js')
def script_js():
    return send_from_directory(BASE_DIR, 'script.js')

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or no selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Get file metadata
    file_size = os.path.getsize(filepath)
    _, extension = os.path.splitext(filepath)
    file_type = extension.lower()

    # Add document to database with attribution
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO documents (filename, file_path, file_size, file_type, uploaded_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, filepath, file_size, file_type, current_user.id))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "File already exists"}), 409
    finally:
        conn.close()

    # Check if this file requires OCR processing
    if file_type == '.pdf':
        # For PDFs, check if regular loader will work or if OCR is needed
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            has_content = any(len(doc.page_content.strip()) > 0 for doc in docs)

            if not has_content:
                # Mark OCR as in progress
                global ocr_processing_status
                ocr_processing_status[filename] = "processing"

                # Start OCR processing asynchronously
                def async_ocr_process():
                    success = rag_pipeline.add_document(filepath)
                    # Update document processing status
                    conn = get_db_connection()
                    conn.execute('''
                        UPDATE documents
                        SET is_processed = ?, processing_status = ?
                        WHERE file_path = ?
                    ''', (success, "completed" if success else "failed", filepath))
                    conn.commit()
                    conn.close()
                    # Mark OCR as done
                    ocr_processing_status[filename] = "done"
                import threading
                threading.Thread(target=async_ocr_process, daemon=True).start()

                # Return immediately with OCR status
                return jsonify({
                    "success": f"File '{filename}' uploaded. OCR processing started asynchronously.",
                    "requires_ocr": True,
                    "filename": filename,
                    "document_id": get_document_id_by_path(filepath)
                }), 200
        except Exception as e:
            print(f"Error checking PDF content: {e}")

    # For non-PDF files or PDFs that don't need OCR, process normally
    success = rag_pipeline.add_document(filepath)

    # Update document processing status
    conn = get_db_connection()
    conn.execute('''
        UPDATE documents
        SET is_processed = ?, processing_status = ?
        WHERE file_path = ?
    ''', (success, "completed" if success else "failed", filepath))
    conn.commit()
    conn.close()

    if success:
        return jsonify({
            "success": f"File '{filename}' uploaded and processed.",
            "document_id": get_document_id_by_path(filepath)
        }), 200
    else:
        return jsonify({"error": f"Could not parse content from '{filename}'. Please try a different file."}), 400

def get_document_id_by_path(filepath):
    """Helper function to get document ID by file path"""
    conn = get_db_connection()
    doc = conn.execute('SELECT id FROM documents WHERE file_path = ?', (filepath,)).fetchone()
    conn.close()
    return doc['id'] if doc else None

@app.route('/api/live_analyze', methods=['POST'])
def live_analyze():
    data = request.get_json()
    draft_content = data.get('draft_content', '')

    if not draft_content.strip():
        return jsonify({"success": True, "analysis": {
            "potential_citations": [], "related_papers": [], "validation_feedback": []
        }})

    result = rag_pipeline.analyze_draft(draft_content)
    return jsonify(result)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question not provided"}), 400

    question = data['question']
    draft_context = data.get('draft_context', '')

    report = rag_pipeline.generate_report(question, draft_context)

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
    return jsonify(dict(stats)) if stats else jsonify({"questions_asked": 0, "reports_generated": 0})

@app.route('/api/ocr_status/<filename>')
def get_ocr_status(filename):
    """Check the status of OCR processing for a file."""
    status = ocr_processing_status.get(filename, "not_found")
    return jsonify({"status": status})

# --- NEW: AUTHENTICATION ROUTES ---
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if user:
        conn.close()
        return jsonify({"error": "Username already exists."}), 409

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
    if not topic:
        return jsonify({"error": "Topic not provided"}), 400
    return jsonify({"success": True, "papers": rag_pipeline.search_concept(topic)})

@app.route('/api/summarize_paper', methods=['POST'])
@login_required
def summarize_paper():
    filepath = request.get_json().get('filepath')
    if not filepath:
        return jsonify({"error": "Filepath not provided"}), 400
    summary = rag_pipeline.summarize_document(filepath)
    if "Error:" in summary:
        return jsonify({"error": summary})
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

# --- NEW: DOCUMENT ORGANIZATION ENDPOINTS ---

@app.route('/api/documents', methods=['GET'])
@login_required
def get_documents():
    """Get all documents with attribution and organization info"""
    conn = get_db_connection()

    # Get documents with uploader information
    documents = conn.execute('''
        SELECT
            d.id, d.filename, d.file_path, d.file_size, d.file_type,
            d.title, d.description, d.uploaded_at, d.is_processed,
            u.username as uploaded_by_username
        FROM documents d
        JOIN users u ON d.uploaded_by = u.id
        ORDER BY d.uploaded_at DESC
    ''').fetchall()

    # Get categories and tags for each document
    result = []
    for doc in documents:
        doc_dict = dict(doc)

        # Get categories for this document
        categories = conn.execute('''
            SELECT c.id, c.name, c.color
            FROM categories c
            JOIN document_categories dc ON c.id = dc.category_id
            WHERE dc.document_id = ?
        ''', (doc['id'],)).fetchall()
        doc_dict['categories'] = [dict(cat) for cat in categories]

        # Get tags for this document
        tags = conn.execute('''
            SELECT t.id, t.name
            FROM tags t
            JOIN document_tags dt ON t.id = dt.tag_id
            WHERE dt.document_id = ?
        ''', (doc['id'],)).fetchall()
        doc_dict['tags'] = [dict(tag) for tag in tags]

        result.append(doc_dict)

    conn.close()
    return jsonify(result)

@app.route('/api/documents', methods=['POST'])
@login_required
def add_document():
    """Add a new document with metadata"""
    data = request.get_json()
    filename = data.get('filename')
    filepath = data.get('filepath')
    file_size = data.get('file_size', 0)
    file_type = data.get('file_type', '')
    title = data.get('title', '')
    description = data.get('description', '')

    if not filename or not filepath:
        return jsonify({"error": "Filename and filepath are required"}), 400

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO documents (filename, file_path, file_size, file_type, title, description, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, filepath, file_size, file_type, title, description, current_user.id))
        conn.commit()
        return jsonify({"success": True, "message": "Document added successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Document already exists"}), 409
    finally:
        conn.close()

@app.route('/api/documents/<int:doc_id>', methods=['PUT'])
@login_required
def update_document(doc_id):
    """Update document metadata"""
    data = request.get_json()
    title = data.get('title')
    description = data.get('description')

    conn = get_db_connection()
    try:
        conn.execute('''
            UPDATE documents
            SET title = ?, description = ?
            WHERE id = ? AND uploaded_by = ?
        ''', (title, description, doc_id, current_user.id))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/categories', methods=['GET'])
@login_required
def get_categories():
    """Get all categories"""
    conn = get_db_connection()
    categories = conn.execute('''
        SELECT c.*, u.username as created_by_username
        FROM categories c
        JOIN users u ON c.created_by = u.id
        ORDER BY c.name
    ''').fetchall()
    conn.close()
    return jsonify([dict(cat) for cat in categories])

@app.route('/api/categories', methods=['POST'])
@login_required
def create_category():
    """Create a new category"""
    data = request.get_json()
    name = data.get('name')
    description = data.get('description', '')
    color = data.get('color', '#3B82F6')

    if not name:
        return jsonify({"error": "Category name is required"}), 400

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO categories (name, description, color, created_by)
            VALUES (?, ?, ?, ?)
        ''', (name, description, color, current_user.id))
        conn.commit()
        return jsonify({"success": True, "message": "Category created successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Category already exists"}), 409
    finally:
        conn.close()

@app.route('/api/documents/<int:doc_id>/categories', methods=['POST'])
@login_required
def add_document_to_category(doc_id):
    """Add a document to a category"""
    data = request.get_json()
    category_id = data.get('category_id')

    if not category_id:
        return jsonify({"error": "Category ID is required"}), 400

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO document_categories (document_id, category_id, added_by)
            VALUES (?, ?, ?)
        ''', (doc_id, category_id, current_user.id))
        conn.commit()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Document is already in this category"}), 409
    finally:
        conn.close()

@app.route('/api/documents/<int:doc_id>/categories/<int:category_id>', methods=['DELETE'])
@login_required
def remove_document_from_category(doc_id, category_id):
    """Remove a document from a category"""
    conn = get_db_connection()
    try:
        conn.execute('''
            DELETE FROM document_categories
            WHERE document_id = ? AND category_id = ?
        ''', (doc_id, category_id))
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/tags', methods=['GET'])
@login_required
def get_tags():
    """Get all tags"""
    conn = get_db_connection()
    tags = conn.execute('''
        SELECT t.*, u.username as created_by_username
        FROM tags t
        JOIN users u ON t.created_by = u.id
        ORDER BY t.name
    ''').fetchall()
    conn.close()
    return jsonify([dict(tag) for tag in tags])

@app.route('/api/tags', methods=['POST'])
@login_required
def create_tag():
    """Create a new tag"""
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({"error": "Tag name is required"}), 400

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO tags (name, created_by)
            VALUES (?, ?)
        ''', (name, current_user.id))
        conn.commit()
        return jsonify({"success": True, "message": "Tag created successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Tag already exists"}), 409
    finally:
        conn.close()

@app.route('/api/documents/<int:doc_id>/tags', methods=['POST'])
@login_required
def add_tag_to_document(doc_id):
    """Add a tag to a document"""
    data = request.get_json()
    tag_id = data.get('tag_id')

    if not tag_id:
        return jsonify({"error": "Tag ID is required"}), 400

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO document_tags (document_id, tag_id, added_by)
            VALUES (?, ?, ?)
        ''', (doc_id, tag_id, current_user.id))
        conn.commit()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Document already has this tag"}), 409
    finally:
        conn.close()

@app.route('/api/documents/search', methods=['GET'])
@login_required
def search_documents():
    """Search documents by title, description, filename, or uploader"""
    query = request.args.get('q', '')
    category_id = request.args.get('category_id')
    tag_id = request.args.get('tag_id')

    if not query and not category_id and not tag_id:
        return jsonify([])

    conn = get_db_connection()

    # Build dynamic query
    sql = '''
        SELECT DISTINCT
            d.id, d.filename, d.file_path, d.file_size, d.file_type,
            d.title, d.description, d.uploaded_at, d.is_processed,
            u.username as uploaded_by_username
        FROM documents d
        JOIN users u ON d.uploaded_by = u.id
        WHERE 1=1
    '''

    params = []

    if query:
        sql += ''' AND (
            d.title LIKE ? OR
            d.description LIKE ? OR
            d.filename LIKE ? OR
            u.username LIKE ?
        )'''
        search_term = f'%{query}%'
        params.extend([search_term] * 4)

    if category_id:
        sql += ''' AND d.id IN (
            SELECT document_id FROM document_categories WHERE category_id = ?
        )'''
        params.append(category_id)

    if tag_id:
        sql += ''' AND d.id IN (
            SELECT document_id FROM document_tags WHERE tag_id = ?
        )'''
        params.append(tag_id)

    sql += ' ORDER BY d.uploaded_at DESC'

    documents = conn.execute(sql, params).fetchall()

    # Add categories and tags to results
    result = []
    for doc in documents:
        doc_dict = dict(doc)

        # Get categories for this document
        categories = conn.execute('''
            SELECT c.id, c.name, c.color
            FROM categories c
            JOIN document_categories dc ON c.id = dc.category_id
            WHERE dc.document_id = ?
        ''', (doc['id'],)).fetchall()
        doc_dict['categories'] = [dict(cat) for cat in categories]

        # Get tags for this document
        tags = conn.execute('''
            SELECT t.id, t.name
            FROM tags t
            JOIN document_tags dt ON t.id = dt.tag_id
            WHERE dt.document_id = ?
        ''', (doc['id'],)).fetchall()
        doc_dict['tags'] = [dict(tag) for tag in tags]

        result.append(doc_dict)

    conn.close()
    return jsonify(result)

# --- Main Execution ---
if __name__ == '__main__':
    init_db()
    pathway_thread = threading.Thread(target=run_pathway_pipeline, daemon=True)
    pathway_thread.start()
    app.run(debug=True, use_reloader=False)

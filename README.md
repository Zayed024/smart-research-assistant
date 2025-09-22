# Smart Research Assistant

An AI-powered web application that helps researchers and students with academic writing, document analysis, and knowledge management. Built with Flask, Pathway, and LangChain for real-time document processing and RAG (Retrieval-Augmented Generation) capabilities.

## 🚀 Features

### Core Functionality
- **Real-time Draft Analysis**: Get instant feedback on your writing with AI-powered analysis
- **Document Upload & Processing**: Support for PDF, TXT, DOCX, and image files with OCR
- **Q&A System**: Ask questions about your documents and get contextual answers
- **Citation Detection**: Automatically identify potential citations and missing references
- **Live Data Streaming**: Real-time document processing with Pathway

### Document Management
- **User Authentication**: Secure login/registration system
- **Document Organization**: Categorize and tag documents for better organization
- **Search & Filter**: Advanced search with category and tag filtering
- **Personal Library**: Save and manage your research documents
- **Usage Statistics**: Track questions asked and reports generated

### Advanced Features
- **Learning Modules**: Concept search and document summarization
- **Export Reports**: Generate formatted reports of your analysis
- **Q&A History**: Keep track of all your questions and answers
- **Multi-format Support**: Handle various document types with intelligent parsing

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework with session management
- **Flask-Login**: User authentication and session handling
- **SQLite**: Database for user data and document metadata
- **Pathway**: Real-time data processing and streaming
- **LangChain**: RAG implementation and document processing
- **Sentence Transformers**: Text embeddings and similarity search
- **FAISS**: Vector database for efficient similarity search

### Frontend
- **HTML5/CSS3**: Modern responsive interface
- **JavaScript (ES6+)**: Interactive user interface
- **Tailwind CSS**: Utility-first CSS framework
- **Font**: Inter font family for clean typography

### Document Processing
- **PyPDF2**: PDF text extraction
- **PyPDFium2**: Advanced PDF processing
- **Pytesseract**: OCR for image-based PDFs
- **Docx2txt**: Word document processing
- **Pillow**: Image processing for OCR

## 📋 Prerequisites

- Python 3.11 
- pip package manager
- Tesseract OCR (for image processing)
- Poppler (for PDF processing)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd smart-research-assistant
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Create .env file
cp .env.example .env
# Edit .env with your API keys (if needed)
```

5. **Initialize database:**
```bash
python -c "from app import init_db; init_db()"
```

6. **Run the application:**
```bash
python app.py
```

7. **Open in browser:**
Navigate to `http://localhost:5000`

## 📖 Usage

### Getting Started

1. **Register/Login**: Create an account or log in to access full features
2. **Upload Documents**: Upload your research papers, PDFs, or documents
3. **Start Writing**: Begin writing in the draft editor
4. **Get Analysis**: Receive real-time feedback and suggestions
5. **Ask Questions**: Use the Q&A system to get contextual answers

### Document Upload

- **Supported formats**: PDF, TXT, DOCX, PNG, JPG, JPEG
- **OCR Support**: Automatic OCR processing for image-based PDFs
- **Batch Processing**: Upload multiple documents for comprehensive analysis
- **Real-time Status**: Track processing status with live updates

### Draft Analysis

- **Live Feedback**: Get instant analysis as you type
- **Citation Detection**: Identify potential citations and missing references
- **Context Awareness**: Analysis based on your uploaded documents
- **Export Options**: Generate formatted reports of your analysis

### Document Organization

- **Categories**: Create and assign categories to documents
- **Tags**: Flexible tagging system for better organization
- **Search**: Advanced search with filters and sorting
- **Library**: Personal collection of saved documents

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Optional: API Keys for enhanced features
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Optional: Custom settings
FLASK_SECRET_KEY=your_secret_key_here
DEBUG=True
```

### Database Schema

The application uses SQLite with the following main tables:
- `users`: User accounts and authentication
- `documents`: Document metadata and file information
- `categories`: Document categorization system
- `tags`: Flexible tagging for documents
- `qa_history`: Question and answer history
- `usage_stats`: Application usage statistics

## 🧪 Testing

### API Testing

Test the API endpoints using curl:

```bash
# Check server status
curl http://localhost:5000/

# Get usage statistics
curl http://localhost:5000/api/stats

# Register a new user
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}'

# Login
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}'
```

### Frontend Testing

1. Open browser to `http://localhost:5000`
2. Test user registration and login
3. Upload a sample document
4. Try the draft analysis feature
5. Test the Q&A system
6. Explore document organization features

## 📊 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/status` - Authentication status

### Document Management
- `POST /api/upload` - Upload documents
- `GET /api/documents` - List user documents
- `GET /api/documents/search` - Search documents
- `POST /api/documents` - Add document metadata

### Analysis & Q&A
- `POST /api/live_analyze` - Real-time draft analysis
- `POST /api/ask` - Ask questions about documents
- `GET /api/qa_history` - Get Q&A history

### Organization
- `GET /api/categories` - List categories
- `POST /api/categories` - Create category
- `GET /api/tags` - List tags
- `POST /api/tags` - Create tag

### Learning & Library
- `POST /api/learn` - Concept search
- `POST /api/summarize_paper` - Document summarization
- `GET /api/library` - User library
- `POST /api/library` - Add to library



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



**Happy Researching!** 📚✨

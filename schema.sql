CREATE TABLE IF NOT EXISTS usage_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1), -- Enforce a single row
    questions_asked INTEGER NOT NULL DEFAULT 0,
    reports_generated INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);

-- Enhanced document management with attribution and metadata
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    file_size INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    title TEXT,
    description TEXT,
    uploaded_by INTEGER NOT NULL,
    uploaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_processed BOOLEAN NOT NULL DEFAULT FALSE,
    processing_status TEXT DEFAULT 'pending',
    FOREIGN KEY (uploaded_by) REFERENCES users (id)
);

-- Document categories for organization
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    color TEXT DEFAULT '#3B82F6',
    created_by INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users (id)
);

-- Many-to-many relationship between documents and categories
CREATE TABLE IF NOT EXISTS document_categories (
    document_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    added_by INTEGER NOT NULL,
    added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, category_id),
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories (id) ON DELETE CASCADE,
    FOREIGN KEY (added_by) REFERENCES users (id)
);

-- Document tags for flexible organization
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_by INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users (id)
);

-- Many-to-many relationship between documents and tags
CREATE TABLE IF NOT EXISTS document_tags (
    document_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    added_by INTEGER NOT NULL,
    added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, tag_id),
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE,
    FOREIGN KEY (added_by) REFERENCES users (id)
);

-- Keep legacy library table for backward compatibility (optional)
CREATE TABLE IF NOT EXISTS library (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id),
    UNIQUE(user_id, file_path)
);

-- Q&A history per user
CREATE TABLE IF NOT EXISTS qa_history (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT,
    draft_context TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
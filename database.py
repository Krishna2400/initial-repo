import sqlite3

# Database connection and setup
def setup_database():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    
    # Table for storing user conversations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        query TEXT,
        response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Table for storing unhandled queries
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS unhandled_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

def log_conversation(user_id, query, response):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO conversations (user_id, query, response) VALUES (?, ?, ?)', (user_id, query, response))
    conn.commit()
    conn.close()

def log_unhandled_query(query):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO unhandled_queries (query) VALUES (?)', (query,))
    conn.commit()
    conn.close()

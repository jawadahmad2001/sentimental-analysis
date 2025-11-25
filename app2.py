# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import whisper
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from deep_translator import GoogleTranslator
from functools import wraps
import bcrypt
import sqlite3
import re
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_very_secure_secret_key'  # Change this in production

# Load Whisper model
whisper_model = whisper.load_model("small")  # You can use "base" or "tiny" for faster performance

# Set up BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'model/Model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Database setup
def init_db():
    conn = sqlite3.connect('suicide_app.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create analysis history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        original_text TEXT,
        translated_text TEXT,
        prediction TEXT NOT NULL,
        suicidal_prob REAL NOT NULL,
        non_suicidal_prob REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Insert admin user if it doesn't exist
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)", 
                      ('admin', hashed_password, 'admin@example.com', 'admin'))
    
    conn.commit()
    conn.close()

init_db()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin access decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('You do not have permission to access this page')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Text preprocessing function (same as in your BERT model)
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ""

# BERT prediction function
def predict_suicide_risk(text, model, tokenizer, device, max_length=128):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
    # Get prediction probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Get class with highest probability
    _, prediction = torch.max(probs, dim=1)
    
    return {
        'prediction': 'Suicidal' if prediction.item() == 1 else 'Non-suicidal',
        'confidence': probs[0][prediction.item()].item(),
        'suicidal_prob': probs[0][1].item(),
        'non_suicidal_prob': probs[0][0].item()
    }

# Analysis function for detailed prediction explanation
def analyze_predictions(text, prediction_result):
    """Analyze why the model made a particular prediction"""
    # Define suicide indicators
    suicide_indicators = [
        'kill', 'die', 'suicide', 'end', 'pain', 'life', 'anymore', 'want', 'hope', 
        'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad', 
        'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless'
    ]
    
    # Define first-person pronouns
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
    
    # Check for suicide indicators
    suicide_indicators_present = []
    for word in suicide_indicators:
        if word in text.lower().split():
            suicide_indicators_present.append(word)
    
    # Check for first-person pronoun usage
    first_person_count = sum(1 for word in text.lower().split() if word in first_person_pronouns)
    
    # Analyze text length
    text_length = len(text)
    word_count = len(text.split())
    
    return {
        'indicators': suicide_indicators_present,
        'first_person_count': first_person_count,
        'text_length': text_length,
        'word_count': word_count
    }

# Save analysis to database
def save_analysis(user_id, original_text, translated_text, prediction, suicidal_prob, non_suicidal_prob):
    conn = sqlite3.connect('suicide_app.db')
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO analysis_history 
           (user_id, original_text, translated_text, prediction, suicidal_prob, non_suicidal_prob) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, original_text, translated_text, prediction, suicidal_prob, non_suicidal_prob)
    )
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html', logged_in=True, username=session.get('username'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('suicide_app.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, password, role, username FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
            session['user_id'] = user[0]
            session['role'] = user[2]
            session['username'] = user[3]
            flash('Successfully logged in')
            return redirect(url_for('index'))
        
        flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        # Basic validation
        if not all([username, password, email]):
            flash('All fields are required')
            return render_template('register.html')
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        try:
            conn = sqlite3.connect('suicide_app.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                (username, hashed_password, email)
            )
            conn.commit()
            conn.close()
            
            flash('Account created successfully, please login')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists')
    
    return render_template('register.html')

@app.route('/transcribe', methods=['POST'])
@login_required
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['file']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    try:
        # Transcribe Urdu audio
        result = whisper_model.transcribe(audio_path, language="ur")
        urdu_text = result['text']

        # Translate Urdu to English
        translated_text = GoogleTranslator(source='ur', target='en').translate(urdu_text)

        # Make prediction with BERT
        prediction_result = predict_suicide_risk(translated_text, model, tokenizer, device)
        
        # Analyze the prediction
        analysis_result = analyze_predictions(translated_text, prediction_result)
        
        # Save analysis to database
        save_analysis(
            session['user_id'], 
            urdu_text, 
            translated_text, 
            prediction_result['prediction'],
            prediction_result['suicidal_prob'],
            prediction_result['non_suicidal_prob']
        )

        return jsonify({
            "urdu_text": urdu_text,
            "translated_text": translated_text,
            "prediction": prediction_result['prediction'],
            "confidence": [prediction_result['non_suicidal_prob'], prediction_result['suicidal_prob']],
            "analysis": analysis_result
        })

    finally:
        # Clean up temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.route('/analyze_text', methods=['POST'])
@login_required
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Make prediction with BERT
    prediction_result = predict_suicide_risk(text, model, tokenizer, device)
    
    # Analyze the prediction
    analysis_result = analyze_predictions(text, prediction_result)
    
    # Save analysis to database (no original Urdu text in this case)
    save_analysis(
        session['user_id'], 
        '', 
        text, 
        prediction_result['prediction'],
        prediction_result['suicidal_prob'],
        prediction_result['non_suicidal_prob']
    )
    
    return jsonify({
        "translated_text": text,
        "prediction": prediction_result['prediction'],
        "confidence": [prediction_result['non_suicidal_prob'], prediction_result['suicidal_prob']],
        "suicidal_prob": prediction_result['suicidal_prob'],
        "non_suicidal_prob": prediction_result['non_suicidal_prob'],
        "analysis": analysis_result
    })

@app.route('/admin')
@login_required
@admin_required
def admin_panel():
    conn = sqlite3.connect('suicide_app.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all users
    cursor.execute("SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    
    # Get analysis statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN prediction = 'Suicidal' THEN 1 ELSE 0 END) as suicidal_count,
            SUM(CASE WHEN prediction = 'Non-suicidal' THEN 1 ELSE 0 END) as non_suicidal_count,
            AVG(suicidal_prob) as avg_suicidal_prob
        FROM analysis_history
    """)
    stats = cursor.fetchone()
    
    # Get recent analyses
    cursor.execute("""
        SELECT a.*, u.username
        FROM analysis_history a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.created_at DESC
        LIMIT 20
    """)
    analyses = cursor.fetchall()
    
    conn.close()
    
    return render_template('admin.html', 
                          users=users, 
                          stats=stats, 
                          analyses=analyses)

@app.route('/user_history')
@login_required
def user_history():
    conn = sqlite3.connect('suicide_app.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get user's analysis history
    cursor.execute("""
        SELECT * FROM analysis_history
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (session['user_id'],))
    analyses = cursor.fetchall()
    
    conn.close()
    
    return render_template('user_history.html', analyses=analyses)

if __name__ == '__main__':
    app.run(debug=True)
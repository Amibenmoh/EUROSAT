from flask import Flask, request, jsonify, session, send_from_directory, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import json
from functools import wraps
import string
import mysql.connector
from mysql.connector import Error
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_super_securisee_12345'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 3600

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "votre_email@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USERNAME)
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1234567890",
            database="eurosat_db",
            autocommit=True
        )
        return connection
    except Error as e:
        print(f"Erreur de connexion à la base de données: {e}")
        return None

MODEL_PATH = "resnet50_fast_model.h5"
INPUT_SIZE = (128, 128)

try:
    model = load_model(MODEL_PATH)
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None

CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

CLASS_DESCRIPTIONS = {
    'AnnualCrop': '🌱 Cultures annuelles : Plantes cultivées et récoltées chaque année (blé, maïs, orge, riz, tournesol). Cycle de vie court.',
    'Forest': "🌳 Forêts : Zones couvertes d'arbres et de végétation dense, naturelles ou plantées. Important pour biodiversité et climat.",
    'HerbaceousVegetation': '🌿 Végétation herbacée : Plantes non ligneuses (herbes, buissons bas, plantes sauvages). Présentes dans prairies ou landes.',
    'Highway': '🛣 Routes et autoroutes : Surfaces asphaltées pour le transport routier (autoroutes, routes principales, carrefours).',
    'Industrial': '🏭 Zones industrielles : Espaces dédiés aux usines, entrepôts, centrales, zones de stockage. Grandes surfaces construites.',
    'Pasture': '🐄 Pâturages : Zones herbeuses utilisées pour nourrir le bétail (vaches, moutons, chevaux). Finalité agricole.',
    'PermanentCrop': '🍇 Cultures permanentes : Plantes cultivées qui repoussent chaque année sans replantation (vignes, vergers, oliviers, caféiers).',
    'Residential': "🏠 Zones résidentielles : Quartiers d'habitations humaines (maisons, immeubles, lotissements).",
    'River': "🌊 Rivières et cours d'eau : Étendues d'eau coulante (rivières, fleuves, canaux). Formes allongées et sinueuses.",
    'SeaLake': "🌊 Mers et lacs : Étendues d'eau statiques (mers, océans, lacs, réservoirs). Grandes surfaces d'eau immobiles."
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def predict_with_model(image_path):
    if model is None:
        return None, 0, None
        
    try:
        img = image.load_img(image_path, target_size=INPUT_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array, verbose=0)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        return pred_class, confidence, preds[0]
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None, 0, None

def process_base64_image(base64_data, filename):
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        return filepath
    except Exception as e:
        print(f"Erreur lors du traitement de l'image: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    image_name VARCHAR(255) NOT NULL,
                    predicted_class VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            if cursor.fetchone()[0] == 0:
                hashed_password = generate_password_hash('admin')
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (%s, %s)",
                    ('admin', hashed_password)
                )
                print("Utilisateur admin créé avec le mot de passe 'admin'")
            
            conn.commit()
            cursor.close()
            conn.close()
            print("Base de données initialisée avec succès")
            
        except Error as e:
            print(f"Erreur lors de l'initialisation de la base de données: {e}")

# NOUVELLE ROUTE AJOUTÉE - CORRECTION DU PROBLÈME
@app.route('/api/user-info', methods=['GET'])
def get_user_info():
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session['user_id'],
                'username': session['username']
            }
        }), 200
    else:
        return jsonify({'authenticated': False}), 200

@app.route('/api/reset-password-by-username', methods=['POST'])
def reset_password_by_username():
    data = request.get_json()
    username = data.get('username')
    new_password = data.get('new_password')
    
    if not username or not new_password:
        return jsonify({'error': "Nom d'utilisateur et nouveau mot de passe requis"}), 400
    
    if len(new_password) < 6:
        return jsonify({'error': 'Le mot de passe doit contenir au moins 6 caractères'}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': "Nom d'utilisateur introuvable"}), 404
        
        hashed_password = generate_password_hash(new_password)
        
        cursor.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (hashed_password, user['id'])
        )
        conn.commit()
        
        return jsonify({
            'message': 'Mot de passe réinitialisé avec succès',
            'username': username
        }), 200
        
    except Error as e:
        print(f"Erreur reset_password_by_username: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    finally:
        conn.close()

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'error': 'Nom d\'utilisateur, email et mot de passe requis'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Le mot de passe doit contenir au moins 6 caractères'}), 400
    
    import re
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        return jsonify({'error': 'Adresse email invalide'}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({'error': 'Nom d\'utilisateur déjà utilisé'}), 400
        
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Adresse email déjà utilisée'}), 400
        
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )
        
        conn.commit()
        return jsonify({'message': 'Compte créé avec succès'}), 201
        
    except Error as e:
        print(f"Erreur register: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Nom d\'utilisateur et mot de passe requis'}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return jsonify({'message': 'Connexion réussie', 'user': {'id': user['id'], 'username': user['username']}}), 200
        else:
            return jsonify({'error': 'Nom d\'utilisateur ou mot de passe incorrect'}), 401
            
    except Error as e:
        print(f"Erreur login: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Déconnexion réussie'}), 200

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files and 'image_data' not in request.json:
        return jsonify({'error': 'Aucune image fournie'}), 400
    
    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'Aucun fichier sélectionné'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            image_data = request.json['image_data']
            filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = process_base64_image(image_data, filename)
            if not filepath:
                return jsonify({'error': 'Erreur lors du traitement de l\'image'}), 400
        
        pred_class, confidence, predictions = predict_with_model(filepath)
        
        if pred_class is None:
            return jsonify({'error': 'Erreur lors de la prédiction'}), 500
        
        class_name = CLASS_NAMES[pred_class]
        class_description = CLASS_DESCRIPTIONS.get(class_name, '')
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (user_id, image_name, predicted_class, confidence) VALUES (%s, %s, %s, %s)",
                (session['user_id'], filename, class_name, confidence)
            )
            conn.commit()
            conn.close()
        
        result = {
            'class': class_name,
            'description': class_description,
            'confidence': round(confidence * 100, 2),
            'image_url': f'/uploads/{filename}',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Erreur predict: {e}")
        return jsonify({'error': 'Erreur lors du traitement de l\'image'}), 500

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT image_name, predicted_class, confidence, timestamp 
            FROM predictions 
            WHERE user_id = %s 
            ORDER BY timestamp DESC
        """, (session['user_id'],))
        
        predictions = cursor.fetchall()
        
        for pred in predictions:
            pred['confidence'] = round(pred['confidence'] * 100, 2)
            pred['image_url'] = f'/uploads/{pred["image_name"]}'
            pred['description'] = CLASS_DESCRIPTIONS.get(pred['predicted_class'], '')
        
        return jsonify(predictions), 200
        
    except Error as e:
        print(f"Erreur get_history: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s", (session['user_id'],))
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT predicted_class, COUNT(*) as count 
            FROM predictions 
            WHERE user_id = %s 
            GROUP BY predicted_class 
            ORDER BY count DESC
        """, (session['user_id'],))
        
        class_stats = {}
        for row in cursor.fetchall():
            class_stats[row[0]] = row[1]
        
        return jsonify({
            'total_predictions': total_predictions,
            'class_distribution': class_stats
        }), 200
        
    except Error as e:
        print(f"Erreur get_stats: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
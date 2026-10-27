import os
import sqlite3
import hashlib
import secrets
import json
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, jsonify, render_template, request, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from fedguard_ml import run_federated_round

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.environ.get('FEDGUARD_DB', os.path.join(BASE_DIR, 'fedguard.db'))
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('FEDGUARD_SECRET_KEY', 'dev-only-change-this-secret-before-deployment'),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE', '0') == '1',
    MAX_CONTENT_LENGTH=32 * 1024,
)


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute('PRAGMA foreign_keys = ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            institution TEXT NOT NULL,
            role TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            round_number INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            loss REAL NOT NULL,
            epsilon REAL NOT NULL,
            noise REAL NOT NULL,
            previous_hash TEXT NOT NULL,
            model_hash TEXT NOT NULL,
            block_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            model_state BLOB,
            dp_delta REAL NOT NULL DEFAULT 0.00001,
            steps_per_client INTEGER NOT NULL DEFAULT 0,
            UNIQUE(user_id, round_number)
        );
        CREATE INDEX IF NOT EXISTS idx_rounds_user ON rounds(user_id, round_number DESC);
    ''')
    # Upgrade earlier prototype databases in place.
    columns = {row[1] for row in db.execute('PRAGMA table_info(rounds)').fetchall()}
    for name, declaration in (('model_state', 'BLOB'), ('dp_delta', 'REAL NOT NULL DEFAULT 0.00001'), ('steps_per_client', 'INTEGER NOT NULL DEFAULT 0')):
        if name not in columns:
            db.execute(f'ALTER TABLE rounds ADD COLUMN {name} {declaration}')
    db.commit()
    db.close()


def current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return get_db().execute(
        'SELECT id, name, institution, role, email FROM users WHERE id = ?', (user_id,)
    ).fetchone()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            return jsonify(error='Please log in to continue.'), 401
        return fn(*args, **kwargs)
    return wrapper


def user_payload(user):
    return {'id': user['id'], 'name': user['name'], 'institution': user['institution'],
            'role': user['role'], 'email': user['email']}


def dashboard_payload(user_id):
    db = get_db()
    rounds = db.execute('SELECT * FROM rounds WHERE user_id = ? ORDER BY round_number ASC', (user_id,)).fetchall()
    latest = rounds[-1] if rounds else None
    ledger = []
    for row in reversed(rounds):
        ledger.append({
            'block': row['round_number'], 'round': row['round_number'],
            'previous_hash': row['previous_hash'], 'model_hash': row['model_hash'],
            'block_hash': row['block_hash'], 'epsilon': row['epsilon'],
            'created_at': row['created_at'], 'accuracy': row['accuracy'],
            'loss': row['loss'], 'noise': row['noise'],
            'dp_delta': row['dp_delta'], 'steps_per_client': row['steps_per_client']
        })
    return {
        'round': latest['round_number'] if latest else 0,
        'accuracy': latest['accuracy'] if latest else None,
        'loss': latest['loss'] if latest else None,
        'epsilon': latest['epsilon'] if latest else 0,
        'history': [{'accuracy': r['accuracy'], 'loss': r['loss']} for r in rounds],
        'ledger': ledger,
    }


@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    response.headers['Cache-Control'] = 'no-store' if request.path.startswith('/api/') else 'no-cache'
    return response


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/health')
def health():
    return jsonify(status='ok', service='FedGuard API', mode='pytorch-fedavg-dp', data='synthetic-only')


@app.get('/api/me')
def me():
    user = current_user()
    if not user:
        return jsonify(user=None)
    return jsonify(user=user_payload(user), dashboard=dashboard_payload(user['id']))


@app.post('/api/register')
def register():
    data = request.get_json(silent=True) or {}
    name = str(data.get('name', '')).strip()
    institution = str(data.get('institution', '')).strip()
    role = str(data.get('role', 'Hospital Admin')).strip()
    email = str(data.get('email', '')).strip().lower()
    password = str(data.get('password', ''))
    if not name or not institution:
        return jsonify(error='Name and institution are required.'), 400
    if len(name) > 100 or len(institution) > 160:
        return jsonify(error='Name or institution is too long.'), 400
    if len(email) > 254 or not __import__('re').fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        return jsonify(error='Enter a valid email address.'), 400
    if role not in ('Hospital Admin', 'Data Scientist', 'Auditor'):
        return jsonify(error='Choose a valid role.'), 400
    checks = [len(password) >= 8, any(c.isupper() for c in password), any(c.isdigit() for c in password), any(not c.isalnum() for c in password)]
    if len(password) < 8 or sum(checks) < 3:
        return jsonify(error='Password must be 8+ characters and include at least 3 of: uppercase, number, symbol.'), 400
    db = get_db()
    try:
        cur = db.execute('INSERT INTO users(name,institution,role,email,password_hash,created_at) VALUES(?,?,?,?,?,?)',
                         (name, institution, role, email, generate_password_hash(password), datetime.now(timezone.utc).isoformat()))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify(error='An account with this email already exists. Please log in.'), 409
    session.clear()
    session['user_id'] = cur.lastrowid
    user = current_user()
    return jsonify(user=user_payload(user), dashboard=dashboard_payload(user['id'])), 201


@app.post('/api/login')
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get('email', '')).strip().lower()
    password = str(data.get('password', ''))
    user = get_db().execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify(error='Invalid email or password.'), 401
    session.clear()
    session['user_id'] = user['id']
    user = current_user()
    return jsonify(user=user_payload(user), dashboard=dashboard_payload(user['id']))


@app.post('/api/logout')
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get('/api/dashboard')
@login_required
def dashboard():
    user = current_user()
    return jsonify(dashboard=dashboard_payload(user['id']))


@app.post('/api/rounds')
@login_required
def start_round():
    data = request.get_json(silent=True) or {}
    try:
        noise = float(data.get('noise', 1.0))
    except (TypeError, ValueError):
        return jsonify(error='Noise multiplier must be a number.'), 400
    if not 0.5 <= noise <= 2.0:
        return jsonify(error='Noise multiplier must be between 0.5 and 2.0.'), 400
    user = current_user()
    db = get_db()
    latest = db.execute('SELECT * FROM rounds WHERE user_id = ? ORDER BY round_number DESC LIMIT 1', (user['id'],)).fetchone()
    round_number = (latest['round_number'] + 1) if latest else 1
    epsilon_before = latest['epsilon'] if latest else 0.0
    previous_state = latest['model_state'] if latest else None

    # Real PyTorch local training on three isolated synthetic clients, then weighted FedAvg.
    try:
        result = run_federated_round(noise, previous_state)
    except Exception:
        app.logger.exception('Federated training round failed')
        return jsonify(error='Federated training failed. Check the server logs.'), 500
    epsilon = epsilon_before + result['epsilon_increment']
    accuracy, loss = float(result['accuracy']), float(result['loss'])
    model_hash = result['model_hash']
    previous_hash = latest['block_hash'] if latest else '0' * 64
    created_at = datetime.now(timezone.utc).isoformat()
    block_body = {
        'user_id': user['id'], 'round': round_number, 'previous_hash': previous_hash,
        'model_hash': model_hash, 'epsilon': round(epsilon, 4), 'noise': round(noise, 4),
        'accuracy': round(accuracy, 6), 'loss': round(loss, 6), 'created_at': created_at,
        'dp_delta': result['delta'], 'steps_per_client': result['steps_per_client'],
    }
    block_hash = hashlib.sha256(json.dumps(block_body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    db.execute('''INSERT INTO rounds(user_id,round_number,accuracy,loss,epsilon,noise,previous_hash,model_hash,block_hash,created_at,model_state,dp_delta,steps_per_client)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
               (user['id'], round_number, accuracy, loss, epsilon, noise, previous_hash, model_hash,
                block_hash, created_at, result['model_state'], result['delta'], result['steps_per_client']))
    db.commit()
    return jsonify(round=round_number, accuracy=accuracy, loss=loss, epsilon=epsilon,
                   epsilon_increment=result['epsilon_increment'], dp_delta=result['delta'],
                   steps_per_client=result['steps_per_client'], clients=result['clients'],
                   training='pytorch-dp-fedavg', model_hash=model_hash, previous_hash=previous_hash,
                   block_hash=block_hash, dashboard=dashboard_payload(user['id'])), 201


@app.get('/api/ledger/verify')
@login_required
def verify_ledger():
    # Recompute every stored block hash and previous-hash link for this account.
    user = current_user()
    rows = get_db().execute('SELECT * FROM rounds WHERE user_id = ? ORDER BY round_number ASC', (user['id'],)).fetchall()
    expected_previous = '0' * 64
    errors = []
    for row in rows:
        if row['previous_hash'] != expected_previous:
            errors.append({'round': row['round_number'], 'issue': 'previous_hash_mismatch'})
        body = {
            'user_id': user['id'], 'round': row['round_number'], 'previous_hash': row['previous_hash'],
            'model_hash': row['model_hash'], 'epsilon': round(row['epsilon'], 4),
            'noise': round(row['noise'], 4), 'accuracy': round(row['accuracy'], 6),
            'loss': round(row['loss'], 6), 'created_at': row['created_at'],
            'dp_delta': row['dp_delta'], 'steps_per_client': row['steps_per_client'],
        }
        calculated = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if calculated != row['block_hash']:
            errors.append({'round': row['round_number'], 'issue': 'block_hash_mismatch'})
        expected_previous = row['block_hash']
    return jsonify(valid=not errors, blocks_checked=len(rows), errors=errors)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error='Request is too large.'), 413


init_db()

if __name__ == '__main__':
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '5000')), debug=os.environ.get('FLASK_DEBUG', '0') == '1')

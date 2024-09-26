from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit
import cv2
import datetime
import mediapipe as mp
import pickle
import numpy as np


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)
current_sign = "No sign detected"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class SignRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sign = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


#Loading the classifiers
with open('one_hand_classifier.pkl', 'rb') as f:
    one_hand_clf = pickle.load(f)

with open('two_hand_classifier.pkl', 'rb') as f:
    two_hand_clf = pickle.load(f)

#Initialising hand landmark drawing settings
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


def process_landmarks(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return np.array([]), [], []
    
    landmarks_array = []
    x_ = []
    y_ = []
    
    for hand_landmarks in multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            landmarks_array.extend([x, y])
            x_.append(x)
            y_.append(y)
    
    return np.array(landmarks_array), x_, y_

def predict_sign(landmarks):
    if len(landmarks) == 0:
        return "No hands detected"
    
    num_hands = len(landmarks) // 42  #21 landmarks * 2 coordinates per hand
    
    try:
        if num_hands == 1:
            prediction = one_hand_clf.predict([landmarks])
        elif num_hands == 2:
            prediction = two_hand_clf.predict([landmarks])
        else:
            return "Invalid number of hands detected"
        
        return prediction[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error"

def gen_frames():
    global current_sign
    camera = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = cv2.flip(frame, 1)
                H, W, _ = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    landmarks, x_, y_ = process_landmarks(results.multi_hand_landmarks)
                    
                    if len(landmarks) > 0:
                        predicted_sign = predict_sign(landmarks)
                        current_sign = predicted_sign  # Update the global variable
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                        
                        if x_ and y_:
                            x1 = max(0, int(min(x_) * W) - 10)
                            y1 = max(0, int(min(y_) * H) - 10)
                            x2 = min(W, int(max(x_) * W) + 10)
                            y2 = min(H, int(max(y_) * H) + 10)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                            cv2.putText(frame, predicted_sign.capitalize(), (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    else:
                        current_sign = "Processing..."
                else:
                    current_sign = "No hands detected"
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/current_sign')
def get_current_sign():
    global current_sign
    return jsonify({'sign': current_sign})

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists')
            return redirect(url_for('signup'))
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/add_sign', methods=['POST'])
@login_required
def add_sign():
    sign = request.json.get('sign')
    if sign:
        new_record = SignRecord(user_id=current_user.id, sign=sign)
        db.session.add(new_record)
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'No sign provided'}), 400

@app.route('/history')
@login_required
def history():
    records = SignRecord.query.filter_by(user_id=current_user.id).order_by(SignRecord.timestamp.desc()).all()
    return render_template('history.html', records=records)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
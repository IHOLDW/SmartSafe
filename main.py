from flask import Flask, render_template, request, redirect, url_for, Response, session, get_flashed_messages, flash
import cv2 as cv
import numpy as np
import time
import os
import smtplib
import mediapipe as mp
import tensorflow as tf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
CLASSES = ['Standing_still', 'Walking']
action_model = tf.lite.Interpreter(model_path = r'models\lstm_tflite.tflite')
tflite_model = tf.lite.Interpreter(model_path = r'models\modelsinception_resnet_v2.tflite')
action_model.allocate_tensors()
tflite_model.allocate_tensors()
input_details_emb = tflite_model.get_input_details()
output_details_emb = tflite_model.get_output_details()
input_details_ac = action_model.get_input_details()
output_details_ac = action_model.get_output_details()
model_path = r'models\face_detection_yunet_2023mar.onnx'
face_detector = cv.FaceDetectorYN.create(model_path, "", (320, 320))

app = Flask(__name__)
app.secret_key = 'f53834515f0583f9f887eee31451b9f543bf8b8bf261aeae'

def create_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT NOT NULL,
        embedding BLOB NOT NULL
    );
    """)
    conn.commit()
    conn.close()
create_database()

# Home Page Route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        embedding = capture_face_embedding()
        if embedding is None:
            flash("Face not detected. Please try again.", "error")
            return redirect(url_for('register'))
        
        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, embedding) VALUES (?, ?, ?)",
                (username, email, embedding.tobytes())
            )
            conn.commit()
            conn.close()
            flash(f"User {username} registered successfully!", "success")
        except Exception as e:
            flash(f"An error occurred: {e}", "error")
            return redirect(url_for('register'))
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/register_feed')
def register_feed():
    return Response(generate_register_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login_feed')
def login_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    if "username" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    username = session["username"]
    return render_template('dashboard.html', username=username)

@app.route('/dashboard_feed')
def dashboard_feed():
    recipient_email = session.get("email")
    username = session.get("username")
    if not recipient_email or not username:
        flash("Session expired. Please log in again.", "error")
        return redirect(url_for('login'))
    return Response(generate_action_feed(recipient_email, username), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_register_feed():
    cap = cv.VideoCapture(0)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    face_detector.setInputSize((frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, faces = face_detector.detect(frame)
        if faces is not None:
            for face in faces:
                bbox = face[:4].astype(int)
                x1, y1, width, height = bbox
                x2, y2 = x1 + width, y1 + height
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

def capture_face_embedding():
    cap = cv.VideoCapture(0)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    face_detector.setInputSize((frame_width, frame_height))
    embedding = None

    for _ in range(150):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture video frame.")
            break

        _, faces = face_detector.detect(frame)
        if faces is not None:
            for face in faces:
                bbox = face[:4].astype(int)
                x1, y1, width, height = bbox
                x2, y2 = x1 + width, y1 + height
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, frame.shape[1]), min(y2, frame.shape[0])
                face_img = frame[y1:y2, x1:x2]

                face_img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
                face_img = cv.resize(face_img, (160, 160))
                face_img = np.float32(face_img) / 255.0
                mean, std = face_img.mean(), face_img.std()
                face_img = ((face_img - mean) / std)
                try:
                    tflite_model.set_tensor(input_details_emb[0]['index'], np.expand_dims(face_img, axis = 0))
                    tflite_model.invoke()
                    embedding = tflite_model.get_tensor(output_details_emb[0]['index'])[0]
                except Exception as e:
                    print(f"Error while predicting embedding: {e}")
                cap.release()
                return embedding

    cap.release()
    print("Error: No face detected.")
    return embedding

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

state = {"match_found": False, "matched_user": None, "stop_stream": False}
def generate_frames():
    cap = cv.VideoCapture(0)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    frame_count = 0
    face_detector.setInputSize((frame_width, frame_height))
    elapsed_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed_time = frame_count / fps
        if not state["match_found"]:
            _, faces = face_detector.detect(frame)
            if faces is not None:
                for face in faces:
                    bbox = face[:4].astype(int)
                    x1, y1, width, height = bbox
                    x2, y2 = x1 + width, y1 + height
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, frame.shape[1]), min(y2, frame.shape[0])
                    face_img = frame[y1:y2, x1:x2]
                    face_img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
                    face_img = cv.resize(face_img, (160, 160))
                    face_img = np.float32(face_img) / 255.0
                    mean, std = face_img.mean(), face_img.std()
                    face_img = ((face_img - mean) / std)
                    tflite_model.set_tensor(input_details_emb[0]['index'], np.expand_dims(face_img, axis = 0))
                    tflite_model.invoke()
                    embedding = tflite_model.get_tensor(output_details_emb[0]['index'])[0]

                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT username, email, embedding FROM users")
                    users = cursor.fetchall()
                    conn.close()
                    for user in users:
                        username, email, registered_embedding_blob = user
                        registered_embedding = np.frombuffer(registered_embedding_blob, dtype=np.float32)

                        similarity = np.dot(embedding, registered_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(registered_embedding))
                        if similarity > 0.7:
                            state["match_found"] = True
                            state["stop_stream"] = True
                            state["matched_user"] = {"username": username, "email": email}
                            send_email(
                                email,
                                f"Face Matched: {username}",
                                f"Welcome to Smart Safe, {username}! You have successfully logged in."
                            )
                            break
                    if state["match_found"]:
                        break
        if state["match_found"]:
            cv.putText(frame, f"Welcome, {state['matched_user']['username']}!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "No Match Detected", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        if state["stop_stream"]:
            break

    cap.release()
    
def generate_action_feed(recipient_email, username):
    vid = cv.VideoCapture(0)
    frame_seq = []
    frame_num = 0
    final_prediction = ''
    alert_sent = False
    last_alert_time = 0 
    if not recipient_email:
        print("Error: No email provided.")
        return
    if not username:
        print("Error: No username provided.")
        return

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        res = pose_model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        if not res.pose_landmarks:
            final_prediction = 'No action'
        if res.pose_landmarks:
            landmarks = res.pose_landmarks.landmark
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            frame_seq.append(pose)
            # mp_drawing.draw_landmarks(
            # frame,
            # res.pose_landmarks,
            # mp_pose.POSE_CONNECTIONS)
            frame_num += 1
        if frame_num == 30:
            frame_seq_np = np.expand_dims(frame_seq, axis=0).astype(np.float32)
            action_model.set_tensor(input_details_ac[0]['index'], frame_seq_np)
            action_model.invoke()
            prediction = action_model.get_tensor(output_details_ac[0]['index'])[0]
            final_prediction = CLASSES[np.argmax(prediction)]
            print(final_prediction)
            frame_seq = []
            frame_num = 0
            if final_prediction == "Walking":
                current_time = time.time()
                if not alert_sent or (current_time - last_alert_time >= 5):
                    alert_sent = True
                    last_alert_time = current_time
                    print("Walking detected. Sending email alert...")
                    try:
                        send_email(
                            recipient_email,
                            f"Activity Alert: {username}",
                            "High activity (Walking) has been detected on your dashboard."
                        )
                        print(f"Email sent successfully to {recipient_email}")
                    except Exception as e:
                        print(f"Error sending email: {e}")
        cv.putText(frame, f"Prediction: {final_prediction}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    vid.release()

@app.route('/after_match')
def after_match():
    if state["match_found"] and state["matched_user"]:
        session["username"] = state["matched_user"]["username"]
        session["email"] = state["matched_user"]["email"]
        flash(f"Welcome {state['matched_user']['username']}! You have successfully logged in.", "success")
        state["match_found"] = False
        state["stop_stream"] = False
        return redirect(url_for("dashboard"))
    else:
        flash("No face match found. Please try again.", "error")
        return redirect(url_for("login"))
    
@app.route('/check_match')
def check_match():
    return {"match_found": state["match_found"]}

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('login'))

def send_email(to_email, subject, body):
    sender_email = 't88065330@gmail.com'
    sender_password = 'fmnf pnee ojxm nozj'
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        print(f"Email sent to {to_email}")
    except smtplib.SMTPException as e:
        print(f"Failed to send email: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False, load_dotenv=False)
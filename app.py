import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)
CORS(app)

# Landmark Index Definitions
m_l, m_r, m_t, m_b = 61, 291, 13, 14
left_b_i, left_b_o = 107, 70
right_b_i, right_b_o = 336, 300
left_e_t, left_e_b = 159, 145
right_e_t, right_e_b = 386, 374
n, c, f = 1, 152, 10

HISTORY_LEN = 20
score_history = []

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])

def compute_confidence_score(landmarks, w, h):
    scores = {}
    nose     = get_point(landmarks, n, w, h)
    chin     = get_point(landmarks, c, w, h)
    forehead = get_point(landmarks, f, w, h)

    upper = nose[1] - forehead[1]  
    lower = chin[1] - nose[1]       
    ratio = upper / (lower + 1e-6)
    scores["head_pitch"] = float(np.clip((ratio - 0.4) / 0.7, 0, 1))

    ml = get_point(landmarks, m_l, w, h)   
    mr = get_point(landmarks, m_r, w, h)  
    mt = get_point(landmarks, m_t, w, h)   
    corner_avg_y = (ml[1] + mr[1]) / 2   
    center_y     = mt[1]                   
    mouth_width  = np.linalg.norm(mr - ml) + 1e-6  
    curve = (center_y - corner_avg_y) / mouth_width
    scores["mouth_curve"] = float(np.clip(0.5 + curve * 3, 0, 1))

    l_brow_i = get_point(landmarks, left_b_i,  w, h)
    l_brow_o = get_point(landmarks, left_b_o,  w, h)
    r_brow_i = get_point(landmarks, right_b_i, w, h)
    r_brow_o = get_point(landmarks, right_b_o, w, h)
    l_eye_t  = get_point(landmarks, left_e_t,     w, h)
    r_eye_t  = get_point(landmarks, right_e_t,    w, h)
    
    left_brow_height  = (l_eye_t[1] - ((l_brow_i[1] + l_brow_o[1]) / 2)) / (h + 1e-6)
    right_brow_height = (r_eye_t[1] - ((r_brow_i[1] + r_brow_o[1]) / 2)) / (h + 1e-6)
    avg_brow = (left_brow_height + right_brow_height) / 2
    scores["brow_height"] = float(np.clip(avg_brow * 30, 0, 1))

    le_t = get_point(landmarks, left_e_t,     w, h)
    le_b = get_point(landmarks, left_e_b,     w, h)
    re_t = get_point(landmarks, right_e_t,    w, h)
    re_b = get_point(landmarks, right_e_b,    w, h)
    
    left_open  = abs(le_b[1] - le_t[1]) / (h + 1e-6)
    right_open = abs(re_b[1] - re_t[1]) / (h + 1e-6)
    avg_eye = (left_open + right_open) / 2
    scores["eye_openness"] = float(np.clip(avg_eye * 50, 0, 1))

    weights = {
        "head_pitch":   0.15,
        "mouth_curve":  0.35,
        "brow_height":  0.15,
        "eye_openness": 0.35,
    }
    final = sum(scores[k] * weights[k] for k in weights)
    return final, scores

def smooth_score(new_score):
    score_history.append(new_score)
    if len(score_history) > HISTORY_LEN:
        score_history.pop(0)
    return sum(score_history) / len(score_history)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process_frame', methods=['POST'])
@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_bytes = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    
    metrics = {
        "score": 0.0,
        "status": "NO FACE DETECTED",
        "sub_scores": {
            "head_pitch": 0.0,
            "mouth_curve": 0.0,
            "brow_height": 0.0,
            "eye_openness": 0.0
        }
    }
    landmarks = []
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = [
                {
                    "x": float(lm.x),
                    "y": float(lm.y)
                }
                for lm in face_landmarks.landmark
            ]

            raw_score, sub_scores = compute_confidence_score(
                face_landmarks.landmark,
                w,
                h
            )

            smoothed = smooth_score(raw_score)

            status_text = (
                "CONFIDENT"
                if smoothed >= 0.62
                else "UNDER-CONFIDENT"
            )

            metrics = {
                "score": round(smoothed, 2),
                "status": status_text,
                "sub_scores": {
                    k: round(v, 2)
                    for k, v in sub_scores.items()
                }
            }

    return jsonify({
        "metrics": metrics,
        "landmarks": landmarks
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
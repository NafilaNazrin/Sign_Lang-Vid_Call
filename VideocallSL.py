import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import time

st.set_page_config(layout="wide")
st.title("Sign Language Video Call - Real-Time")

# 🧪 DEBUG: Model loading
try:
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model']
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 🧪 DEBUG: Before initializing WebRTC
st.info("🔄 Initializing WebRTC Streamer...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
    20: 'space', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'
}

# Session states
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'buffer' not in st.session_state:
    st.session_state.buffer = deque(maxlen=10)
if 'last_added_char' not in st.session_state:
    st.session_state.last_added_char = None
if 'last_char_time' not in st.session_state:
    st.session_state.last_char_time = 0


class SignLangTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        img = image.copy()

        x_, y_, data_aux = [], [], []

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                if x_ and y_:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    data_aux = data_aux[:42]
                    while len(data_aux) < 42:
                        data_aux.append(0)

                    prediction = model.predict([np.asarray(data_aux)])
                    try:
                        predicted_index = int(prediction[0])
                        predicted_character = labels_dict.get(predicted_index, '?')
                    except (ValueError, IndexError):
                        predicted_character = '?'

                    st.session_state.buffer.append(predicted_character)

                    if len(st.session_state.buffer) >= 8 and \
                       st.session_state.buffer.count(st.session_state.buffer[-1]) >= 8:
                        stable_char = st.session_state.buffer[-1]
                        current_time = time.time()

                        if stable_char == "space":
                            if st.session_state.current_word:
                                st.session_state.sentence += st.session_state.current_word + " "
                                st.session_state.current_word = ""
                            st.session_state.last_added_char = None
                        elif stable_char == "del":
                            st.session_state.current_word = st.session_state.current_word[:-1]
                            st.session_state.last_added_char = None
                        else:
                            if stable_char != st.session_state.last_added_char or \
                               (current_time - st.session_state.last_char_time > 0.8):
                                st.session_state.current_word += stable_char
                                st.session_state.last_added_char = stable_char
                                st.session_state.last_char_time = current_time

        cv2.putText(img, f"Sentence: {st.session_state.sentence}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Current: {st.session_state.current_word}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return img


webrtc_streamer(key="signlang", video_processor_factory=SignLangTransformer)

st.subheader("Translated Text")
st.text_area("Output", 
             value=st.session_state.sentence + " " + st.session_state.current_word, 
             height=100, 
             key="output_text")

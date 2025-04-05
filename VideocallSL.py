import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import time

# Set up the page
st.set_page_config(layout="wide")
st.title("Sign Language Video Call - Real-Time")

# Load the model
try:
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model']
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Define label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
    20: 'space', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'
}

# Define the transformer class for video processing
class SignLangTransformer(VideoTransformerBase):
    def __init__(self):
        self.buffer = deque(maxlen=10)
        self.sentence = ""
        self.current_word = ""
        self.last_added_char = None
        self.last_char_time = 0
        # Instantiate the MediaPipe Hands and drawing utils once
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        img = image.copy()

        data_aux, x_, y_ = [], [], []
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                if x_ and y_:
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))
                    data_aux = data_aux[:42]
                    while len(data_aux) < 42:
                        data_aux.append(0)
                    prediction = model.predict([np.asarray(data_aux)])
                    try:
                        predicted_index = int(prediction[0])
                        predicted_character = labels_dict.get(predicted_index, '?')
                    except Exception:
                        predicted_character = '?'
                    self.buffer.append(predicted_character)
                    if len(self.buffer) >= 8 and self.buffer.count(self.buffer[-1]) >= 8:
                        stable_char = self.buffer[-1]
                        current_time = time.time()
                        if stable_char == "space":
                            if self.current_word:
                                self.sentence += self.current_word + " "
                                self.current_word = ""
                            self.last_added_char = None
                        elif stable_char == "del":
                            self.current_word = self.current_word[:-1]
                            self.last_added_char = None
                        else:
                            if stable_char != self.last_added_char or (current_time - self.last_char_time > 0.8):
                                self.current_word += stable_char
                                self.last_added_char = stable_char
                                self.last_char_time = current_time

        cv2.putText(img, f"Sentence: {self.sentence}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Current: {self.current_word}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return img

# Initialize the transformer instance and store it in session_state
if "transformer_instance" not in st.session_state:
    st.session_state["transformer_instance"] = SignLangTransformer()

transformer_instance = st.session_state["transformer_instance"]

# Start the webrtc_streamer with a unique key
webrtc_streamer(
    key="signlang_video",
    video_processor_factory=lambda: transformer_instance
)

st.subheader("Translated Text")
st.text_area("Output", value=transformer_instance.sentence + " " + transformer_instance.current_word, height=100)

# Optional: Button to test model prediction with random data
if st.button("🔍 Test Model Prediction with Random Data"):
    fake_input = np.random.rand(42).tolist()
    try:
        prediction = model.predict([fake_input])
        st.success(f"✅ Model Prediction Successful: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Model Prediction Failed: {e}")

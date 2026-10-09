import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import numpy as np

# ==========================================
# 1. TEXT-TO-SPEECH (NON-BLOCKING ENGINE)
# ==========================================
class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)  # Speech speed
        self.is_speaking = False

    def _speak_thread(self, text):
        self.is_speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_speaking = False

    def speak(self, text):
        # Only speak if not already speaking to prevent thread locking
        if not self.is_speaking:
            t = threading.Thread(target=self._speak_thread, args=(text,))
            t.daemon = True
            t.start()

# ==========================================
# 2. GESTURE CLASSIFIER (GEOMETRIC RULES)
# ==========================================
def get_finger_states(landmarks):
    """
    Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]
    True = Open/Extended, False = Folded/Closed
    """
    finger_states = []
    
    # Landmark indices for fingertip and the joint below it
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    
    # 1. THUMB (Checked via x-coordinate relative to palm center/IP joint)
    # Note: Assumes right hand facing camera; works well for general open/close
    if landmarks[tips[0]].x < landmarks[pips[0]].x:
        finger_states.append(True)
    else:
        finger_states.append(False)
        
    # 2. 4 FINGERS (Checked via y-coordinate: Tip higher than lower joint)
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[pips[i]].y:
            finger_states.append(True)
        else:
            finger_states.append(False)
            
    return finger_states

def classify_gesture(finger_states, landmarks):
    """
    Maps finger states to common signs/words.
    You can easily add custom gestures here!
    """
    # Unpack states: [Thumb, Index, Middle, Ring, Pinky]
    t, i, m, r, p = finger_states
    
    # "HELLO" / "STOP" - All fingers open
    if finger_states == [True, True, True, True, True]:
        return "HELLO"
    
    # "YES" / "A" gesture - All fingers closed into a fist
    elif finger_states == [False, False, False, False, False]:
        return "YES"
    
    # "PEACE" / "VICTORY" - Index and Middle open only
    elif i and m and not r and not p:
        return "PEACE"
    
    # "I LOVE YOU" (ASL) - Thumb, Index, and Pinky open
    elif t and i and not m and not r and p:
        return "I LOVE YOU"
    
    # "POINING / ONE" - Only Index extended
    elif not t and i and not m and not r and not p:
        return "ONE"
        
    # "OK" - Thumb and Index tips touch while others are open
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_tip = np.array([landmarks[8].x, landmarks[8].y])
    distance = np.linalg.norm(thumb_tip - index_tip)
    if distance < 0.05 and m and r and p:
        return "OK"

    return "LISTENING..."

# ==========================================
# 3. MAIN EXHIBITION LOOP
# ==========================================
def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    speech = SpeechEngine()
    cap = cv2.VideoCapture(0)

    # Cooldown tracking so it doesn't repeat the same word endlessly
    last_spoken_word = ""
    last_spoken_time = 0
    cooldown_seconds = 2.5  # Time before the same word can be spoken again

    print("=== 'HEAR ME' PROJECT RUNNING ===")
    print("Press 'q' in the video window to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read webcam frame.")
            break

        # Flip horizontally for a natural mirror-view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        current_gesture = "NO HAND DETECTED"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw green landmark skeleton on the hand
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

                # Get landmark states & classify
                finger_states = get_finger_states(hand_landmarks.landmark)
                current_gesture = classify_gesture(finger_states, hand_landmarks.landmark)

                # Trigger speech if a recognized gesture is held stably
                if current_gesture not in ["LISTENING...", "NO HAND DETECTED"]:
                    current_time = time.time()
                    
                    # Speak if it's a new word OR if cooldown has passed
                    if (current_gesture != last_spoken_word) or (current_time - last_spoken_time > cooldown_seconds):
                        speech.speak(current_gesture)
                        last_spoken_word = current_gesture
                        last_spoken_time = current_time

        # ==========================================
        # 4. DRAW EXHIBITION UI ON THE VIDEO
        # ==========================================
        # Top banner background
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        
        # Display Translated Text
        cv2.putText(frame, f"TRANSLATION: {current_gesture}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Show the video feed
        cv2.imshow("Hear Me - Sign Language Translator", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
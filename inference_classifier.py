import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# Loading the trained models
with open('one_hand_classifier.pkl', 'rb') as f:
    one_hand_clf = pickle.load(f)

with open('two_hand_classifier.pkl', 'rb') as f:
    two_hand_clf = pickle.load(f)

# Initializing the camera
cap = cv2.VideoCapture(0)  # might need to change index depending on system

# Set up MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hand detection model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create predictions folder if it doesn't exist
if not os.path.exists('predictions'):
    os.makedirs('predictions')

# Dictionary to keep track of image counts for each letter
image_counts = {}

def process_landmarks(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return np.array([])
    
    landmarks_array = []
    for hand_landmarks in multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            landmarks_array.extend([landmark.x, landmark.y])
    
    return np.array(landmarks_array)

def save_prediction_image(frame, predicted_character):
    # Create folder for the letter if it doesn't exist
    letter_folder = os.path.join('predictions', predicted_character)
    if not os.path.exists(letter_folder):
        os.makedirs(letter_folder)
    
    # Get the current count for this letter
    count = image_counts.get(predicted_character, 0)
    
    # Save the image
    image_path = os.path.join(letter_folder, f"{count}.jpg")
    cv2.imwrite(image_path, frame)
    
    # Increment the count
    image_counts[predicted_character] = count + 1

# Main loop for processing video frames
while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Convert the frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # If hands are detected in the frame
    if results.multi_hand_landmarks:
        landmarks = process_landmarks(results.multi_hand_landmarks)
        num_hands = len(results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Make a prediction using the appropriate classifier
        if num_hands == 1:
            prediction = one_hand_clf.predict([landmarks])
            predicted_character = prediction[0]  # Directly use the predicted letter
        elif num_hands == 2:
            prediction = two_hand_clf.predict([landmarks])
            predicted_character = prediction[0]  # Directly use the predicted letter
        else:
            predicted_character = "Invalid"

        # Calculate the bounding box for all detected hands
        x_min = min(min(landmark.x for landmark in hand_landmarks.landmark) for hand_landmarks in results.multi_hand_landmarks)
        y_min = min(min(landmark.y for landmark in hand_landmarks.landmark) for hand_landmarks in results.multi_hand_landmarks)
        x_max = max(max(landmark.x for landmark in hand_landmarks.landmark) for hand_landmarks in results.multi_hand_landmarks)
        y_max = max(max(landmark.y for landmark in hand_landmarks.landmark) for hand_landmarks in results.multi_hand_landmarks)

        x1 = max(0, int(x_min * W) - 10)
        y1 = max(0, int(y_min * H) - 10)
        x2 = min(W, int(x_max * W) + 10)
        y2 = min(H, int(y_max * H) + 10)

        # Draw the bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Save the prediction image
        save_prediction_image(frame, predicted_character)

    # Display the processed frame
    cv2.imshow('frame', frame)
    
    # Wait for 1ms and check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
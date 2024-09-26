import cv2
import os
import mediapipe as mp
import pickle

# Setting up mediapipe with liberal settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)


def add_text_with_background(frame, text):

    frame_height, frame_width = frame.shape[:2]
    font_scale = 1
    thickness = 0

    #Getting the width and height of the text
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    #Calculating the coordinates for background rectangle by adding 20px to text height and width
    x_position = (frame_width - text_width)//2
    y_position = frame_height//2
    x,y = x_position, y_position
    bg_rect = [(x, y), (x + text_width + 20, y - text_height - 20)]
    
    #Creating a semi transparent background
    bg = frame.copy()
    cv2.rectangle(bg, bg_rect[0], bg_rect[1], (0,0,0), -1)
    frame = cv2.addWeighted(bg, 0.5, frame, 0.5, 0)
    
    #Adding white text with black border
    cv2.putText(frame, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness*3, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    
    return frame


def countdown(cap):
    for i in range(3, 0, -1):
        for _ in range(30):  #Showing each countdown number for 1 second (30 frames at 30 fps)
            ret, frame = cap.read()
            frame = add_text_with_background(frame, str(i))
            cv2.imshow('frame', frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):  #33ms delay for ~30 fps
                return False
    return True


def process_frame(frame, required_hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        if num_hands == required_hands:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):  #21 landmarks per hand (x+y=42)
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.extend([x, y])
            
            return data_aux
    
    return None



def capture_images_for_letter(cap, letter, required_hands, path_to_images, dataset_size):

    #Creating folder for each letter
    if not os.path.exists(os.path.join(path_to_images, str(letter))):
        os.makedirs(os.path.join(path_to_images, str(letter)))
    
    while True:
        ret, frame = cap.read()
        frame_with_text = add_text_with_background(frame.copy(), f"Press Spacebar to record letter '{letter}'")
        cv2.imshow('frame', frame_with_text)
        if cv2.waitKey(25) == ord(' '): #waits for spacebar
            if countdown(cap):
                break
            else:
                print("Countdown interrupted. Press Spacebar to try again.")
    
    letter_data = []
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        hand_data = process_frame(frame, required_hands)
        
        if hand_data:
            cv2.imwrite(os.path.join(path_to_images, str(letter), f'{counter}.jpg'), frame)
            letter_data.append(hand_data)
            counter += 1
        
        #Create a copy of the frame for display purposes
        display_frame = frame.copy()
        status_text = f"Captured: {counter}/{dataset_size}"
        display_frame = add_text_with_background(display_frame, status_text)
        cv2.imshow('frame', display_frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    return letter_data




def create_dataset():
    
    path_to_images = './data'
    if not os.path.exists(path_to_images):
        os.makedirs(path_to_images)

    one_hand_classes = ['c','i','l','o','u','v']
    two_hand_classes = ['a','b','d','e','f','g','h','j','k','m','n','p','q','r','s','t','w','x','y','z']

    dataset_size = 100

    #Initializing the camera
    cap = cv2.VideoCapture(0) #might need to change index depending on system

    data = []
    labels = []

    for letter in (one_hand_classes + two_hand_classes):
        print(letter)
        required_hands = 1 if letter in one_hand_classes else 2
        letter_data = capture_images_for_letter(cap, letter, required_hands, path_to_images, dataset_size)
        data.extend(letter_data)
        labels.extend([letter] * len(letter_data))

    cap.release()
    cv2.destroyAllWindows()

    #Saving the data
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)


if __name__ == "__main__":
    create_dataset()
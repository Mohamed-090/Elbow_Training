import cv2
import mediapipe as mp
import math
import numpy as np
import time   


# Initialization
step1_completed = False
step2_completed = False
step3_completed = False
round_scores = 0

exercise_completed = False 
total_round_scores = 0
round_count = 0
max_rounds = 3   

 
timer = 50
rest_time = 5    
start_time = time.time()   

# Function to calculate angle between three points (shoulder, elbow, hip)
def calculate_angle(a, b, c):
    AB = (b[0] - a[0], b[1] - a[1])
    BC = (c[0] - b[0], c[1] - b[1])
    
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]
    cross_product = AB[0] * BC[1] - AB[1] * BC[0]
    
    angle = math.degrees(math.atan2(cross_product, dot_product))
    if angle < 0:
        angle += 360
    return angle

 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# OpenCV initialization for webcam
cap = cv2.VideoCapture(0)   
while cap.isOpened() and round_count < max_rounds:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image with mediapipe
    results = pose.process(image)
    
    if results.pose_landmarks:
         
        left_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1],
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])
        left_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.shape[1],
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.shape[0])
        left_hip = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1],
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])




        # Calculate angles
        angle_shoulder_elbow_hip = calculate_angle(left_shoulder, left_elbow, left_hip)
        
        # Update step completion based on angle conditions
        if 100 <= angle_shoulder_elbow_hip <= 150 and not step1_completed:
            status_text = "Step 1 SUCCESS"
            step2_completed = True
            step3_completed = False
        elif 30 <= angle_shoulder_elbow_hip <= 70 and step2_completed:
            status_text = "Step 2 SUCCESS"
            step1_completed = True
            step3_completed = True  
        elif angle_shoulder_elbow_hip > 150 and step3_completed:
            status_text = "Step 3 SUCCESS"
            step2_completed = False
            step1_completed = False
            exercise_completed = True   
        else:
            status_text = ""   

        # Handle exercise completion and timer reset
        if exercise_completed:    
            step1_completed = False
            step2_completed = False
            step3_completed = False
            exercise_completed = False
            round_scores += 1
 
         
        elapsed_time = time.time() - start_time
        if elapsed_time >= timer:
            start_time = time.time()           
            total_round_scores  += round_scores 
            round_count += 1
            round_scores=0
 
          #  time.sleep(rest_time) 
          #  rest_text = f"Resting for {rest_time} seconds..."
          #  cv2.putText(frame, rest_text, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display round scores and total score
        round_text = f"Round {round_count } Score: {round_scores}"
        total_text = f"Total Score : {total_round_scores}"
        timer_text = f"Time: {round(timer - elapsed_time)} seconds"
        
        # Print angle (for demonstration)
        angle_text = f"Angle: {angle_shoulder_elbow_hip:.2f} degrees"
        
        # Add a gray background for all text
        cv2.rectangle(frame, (10, 10), (300, 230), (192, 192, 192), -1)  # Rectangle coordinates and color
        
        # Display texts
        cv2.putText(frame, round_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, total_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, timer_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, status_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
     
    # Display frame
    cv2.imshow('Pose Detection', frame)
    
    # Exit loop if 'q' is pressed or all rounds are completed
    if cv2.waitKey(1) & 0xFF == ord('q') or round_count >= max_rounds:
        break

cap.release()
cv2.destroyAllWindows()


while True:
    gray_color = (128, 128, 128)  # RGB values for gray color
    blank_image = np.full((480, 640, 3), gray_color, dtype=np.uint8)

  
    cv2.putText(blank_image, f"Total Score: {total_round_scores}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    # Display the image in a window
    cv2.imshow('Statistics', blank_image)
    



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


import cv2
from fer import FER

# Initialize the webcam and the facial emotion recognition model
cap = cv2.VideoCapture(0)
emotion_detector = FER()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect emotions in the frame
    emotion_results = emotion_detector.detect_emotions(frame)

    # Display the emotions on the frame
    for face in emotion_results:
        emotions = face['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Facial Emotion Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

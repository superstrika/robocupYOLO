from ultralytics import YOLO
import cv2

# 1. Load your optimized ONNX model
model = YOLO('best.onnx', task='detect')

# 2. Initialize your camera (0 is usually the default system webcam)
cap = cv2.VideoCapture(0)

print("Starting camera feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # 3. Run detection.
    # PRO-TIP: Lowering 'imgsz' to 320 makes it run significantly faster on a Pi 4!
    results = model(frame, imgsz=640, conf=0.4)
    
    # 4. Draw bounding boxes on the live frame
    annotated_frame = results[0].plot()
    
    # Display the live window
    cv2.imshow('Orange Ball Tracker', annotated_frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("C:/Users/Hp/OneDrive/Desktop/Project/runs/detect/train6/weights/best.pt")

# Open the video file
video_path = "plane.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video writer setup to save output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output_vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(source=frame, save=False)

    # Draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            if cls == 0:  # Assuming class 0 is the 'Jet' class
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Jet: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

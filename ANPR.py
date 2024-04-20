#THIS WAS MORE TESTING YOU CAN IGNORE
import cv2
import pytesseract
from roboflow import Roboflow

# Set up the Roboflow API
api_key = "" #roboflow API key
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform Roboflow inference
    predictions_json = model.predict(frame, confidence=40, overlap=30).json()
    predictions = predictions_json["predictions"]

    # Draw bounding boxes and perform OCR
    for bbox in predictions:
        x0 = bbox['x'] - bbox['width'] / 2
        x1 = bbox['x'] + bbox['width'] / 2
        y0 = bbox['y'] - bbox['height'] / 2
        y1 = bbox['y'] + bbox['height'] / 2

        # Draw bounding box
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        # Perform OCR on the license plate
        license_plate = frame[int(y0):int(y1), int(x0):int(x1)]
        text = pytesseract.image_to_string(license_plate, config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        # Draw text above the bounding box
        cv2.putText(frame, text.strip(), (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('License Plate Recognition', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

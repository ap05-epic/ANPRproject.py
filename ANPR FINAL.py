#RUN THIS FILE FOR WHAT MY FINAL PRODUCT WOULD HAVB LOOKED LIKE AFTER OU SETUP YOUR API KEYS
import cv2
import time
from roboflow import Roboflow
from google.cloud import vision

# Set up the Roboflow API
api_key = "" #roboflow API key
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(3).model

# Set up the Google Cloud Vision API
vision_client = vision.ImageAnnotatorClient.from_service_account_json("C:\\googleAccountAPI.json")
#I forgot how to get the API .json but its somehwere in the google cloud panel related to image to text

# Initialize webcam
cap = cv2.VideoCapture(0)

min_detection_duration = 2  # In seconds
license_plates = {}
last_printed_license_plate = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    current_time = time.time()

    # Perform Roboflow inference
    predictions_json = model.predict(frame, confidence=40, overlap=30).json()
    predictions = predictions_json["predictions"]

    detected_license_plates = []

    # Draw bounding boxes and perform OCR
    for bbox in predictions:
        x0 = bbox['x'] - bbox['width'] / 2
        x1 = bbox['x'] + bbox['width'] / 2
        y0 = bbox['y'] - bbox['height'] / 2
        y1 = bbox['y'] + bbox['height'] / 2

        # Draw bounding box
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        # Crop license plate and remove extraneous text
        license_plate = frame[int(y0):int(y1), int(x0):int(x1)]
        cropped_license_plate = license_plate[int(0.3 * license_plate.shape[0]):int(0.7 * license_plate.shape[0]), :]

        # Perform OCR on the license plate using Google Cloud Vision API
        _, encoded_image = cv2.imencode('.png', cropped_license_plate)
        image_content = encoded_image.tobytes()
        image = vision.Image(content=image_content)
        response = vision_client.text_detection(image=image)
        text = response.text_annotations[0].description if response.text_annotations else ""

        # Draw text above the bounding box
        cv2.putText(frame, text.strip(), (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        detected_license_plates.append(text.strip())

    # Update license plates dictionary and check for stable detections
    for detected_plate in detected_license_plates:
        if detected_plate not in license_plates:
            license_plates[detected_plate] = current_time
        else:
            if current_time - license_plates[detected_plate] >= min_detection_duration and detected_plate != last_printed_license_plate:
                print(f"Stable detection: {detected_plate}")
                last_printed_license_plate = detected_plate

    # Display the resulting frame
    cv2.imshow('License Plate Recognition', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

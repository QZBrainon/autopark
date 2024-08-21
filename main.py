import cv2
import requests
import numpy as np
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Token for Plate Recognizer
API_TOKEN = os.getenv('API_TOKEN')

# Haarcascade model path
harcascade = 'models/haarcascade_russian_plate_number.xml'

# Minimum area to consider for detection
min_area = 500

# Debouncing variables
DEBOUNCE_TIME = 5  # Time in seconds
last_sent_time = 0

def initialize_camera(stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise Exception("Could not open video stream")
    return cap

def send_to_api(image):
    try:
        # Convert the image to JPEG format
        success, img_encoded = cv2.imencode('.jpg', image)
        if not success:
            print("Failed to encode image")
            return None

        # Prepare the API request
        files = {'upload': ('plate.jpg', img_encoded.tobytes(), 'image/jpeg')}
        headers = {'Authorization': f"Token {API_TOKEN}"}
        response = requests.post('https://api.platerecognizer.com/v1/plate-reader/', files=files, headers=headers)
        
        # Check if request was successful
        response.raise_for_status()
        result = response.json()

        # Check if results list is empty
        plates = result.get('results', [])
        if not plates:
            print("No plates detected")
            return None

        # Extract plate information from the first detected plate
        plate = plates[0].get('plate')
        print("Detected Plate:", plate)
        return plate

    except requests.RequestException as e:
        print(f"API request error: {e}")
        return None

def main():
    # Replace with the actual URL provided by the streaming app
    stream_url = 'http://192.168.100.24:4747/video'

    # Initialize video capture from the stream URL
    cap = initialize_camera(stream_url)

    global last_sent_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Initialize the plate cascade classifier
        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect plates in the image
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        plate_detected = False

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                # Draw rectangle around the plate
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                # Extract the region of interest (ROI)
                img_roi = frame[y: y + h, x: x + w]
                cv2.imshow("ROI", img_roi)

                # Set flag to indicate plate detection
                plate_detected = True

                # Process and send the image to the API if debounce time has passed
                current_time = time.time()
                if current_time - last_sent_time > DEBOUNCE_TIME and plate_detected:
                    plate = send_to_api(img_roi)
                    last_sent_time = current_time
                    # Optionally, you can display or use the plate information as needed
                    print(f"Plate {plate} processed")

                    # Clear the ROI after processing
                    img_roi = None
                    break

        # Display the result
        cv2.imshow("Result", frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # # Save the image if 's' key is pressed
        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     # Check if ROI exists
        #     if 'img_roi' in locals() and img_roi is not None:
        #         cv2.imwrite("plates/scanned_img_" + str(int(time.time())) + ".jpg", img_roi)
        #         cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        #         cv2.putText(frame, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        #         cv2.imshow("Results", frame)
        #         cv2.waitKey(500)
        #         img_roi = None

        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

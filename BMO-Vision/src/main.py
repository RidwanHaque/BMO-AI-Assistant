# File: /BMO-Vision/BMO-Vision/src/main.py

import cv2
import yaml
import logging
from vision import Vision
from utils import setup_database

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config('config/config.yaml')

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database connection
    db_connection = setup_database(config['database'])

    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logging.error("Could not open camera.")
        return

    # Initialize vision processing
    vision = Vision(camera, db_connection)

    # Main loop for processing
    while True:
        ret, frame = camera.read()
        if not ret:
            logging.error("Failed to capture frame.")
            break
        
        # Process the frame for facial and object recognition
        vision.process_frame(frame)

        # Display the resulting frame
        cv2.imshow('BMO Vision', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()
    db_connection.close()

if __name__ == "__main__":
    main()
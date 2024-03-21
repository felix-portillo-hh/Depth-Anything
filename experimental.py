import cv2
import numpy as np 

def capture_background(capture):
    # Capture a single frame
    _, background_frame = capture.read()
    background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    return background_gray

# Start capturing frames from the webcam
webcam_capture = cv2.VideoCapture(0)

# Prompt user to capture the background
input("Press Enter to capture background...")
background_gray = capture_background(webcam_capture)

while True:
    # Capture frame from webcam
    _, frame = webcam_capture.read()
    
    # Convert frame to grayscale for comparison
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between background and current frame
    diff_frame = cv2.absdiff(gray_frame, background_gray)
    
    # Apply a threshold to highlight differences
    _, thresholded_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out smaller contours based on a minimum threshold area
    min_contour_area = 70000  # Adjust this threshold as needed
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create an empty mask for dilation
    mask = np.zeros_like(thresholded_frame)

    # Draw contours on the mask
    cv2.drawContours(mask, large_contours, -1, 255, thickness=cv2.FILLED)

    # Apply dilation to merge nearby regions
    kernel = np.ones((30, 30), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in the dilated mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the resulting frame with contours
    cv2.imshow('Motion Detection with Contours', frame)
    
    # Wait for Esc key to stop the program  
    k = cv2.waitKey(30) & 0xff
    if k == 27:  
        break

# Release the webcam and close all windows
webcam_capture.release()
cv2.destroyAllWindows()

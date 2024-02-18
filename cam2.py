import cv2

# Try to open cameras from index 0 to 9
for i in range(10):
    cap = cv2.VideoCapture(i)

    # Check if the camera opened successfully
    if cap.isOpened():
        print(f"Camera at index {i} opened successfully.")
        break
    else:
        print(f"Error: Could not open camera at index {i}.")

# Your code for camera processing goes here

# Release the camera when done
cap.release()

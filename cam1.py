import cv2

# Open camera at index 0
cap_0 = cv2.VideoCapture(0)
if not cap_0.isOpened():
    print("Error: Could not open camera at index 0.")
    exit()

# Open camera at index 1
cap_1 = cv2.VideoCapture(2)
if not cap_1.isOpened():
    print("Error: Could not open camera at index 1.")
    cap_0.release()  # Release the first camera
    exit()

while True:
    # Read frames from the first camera
    ret_0, frame_0 = cap_0.read()
    if not ret_0:
        print("Error: Could not read frame from camera at index 0.")
        break

    # Read frames from the second camera
    ret_1, frame_1 = cap_1.read()
    if not ret_1:
        print("Error: Could not read frame from camera at index 1.")
        break

    # Display frames
    cv2.imshow("Camera 0", frame_0)
    cv2.imshow("Camera 1", frame_1)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close windows
cap_0.release()
cap_1.release()
cv2.destroyAllWindows()

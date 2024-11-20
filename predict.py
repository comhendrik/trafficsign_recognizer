import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

model = load_model('traffic_sign_model.keras')
labels = ["20", "30", "50", "60", "70", "80",
          "80 lifted", "100", "120", "No overtaking", "No overtaking by trucks",
          "Priority road", "Main road", "Give way", "Stop",
          "No entry", "No trucks", "No entry one-way street",
          "Attention", "Sharp left turn", "Sharp right turn", "Multiple curves", "Bump", "Ice hazard",
          "Narrowing ahead", "Construction zone", "Traffic lights", "Pedestrians", "Children", "Bicycles",
          "Frost warning", "Wildlife crossing", "Unlimited!!!", "No right turn", "No left turn",
          "Go straight", "Go straight and turn right", "Go straight and turn left", "Pass on the right", "Pass on the left",
          "Roundabout", "End of no overtaking", "End of truck ban"]  # Klassenlabels

def preprocess(imgBGR):
    # Convert the image to HSV color space
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

    # Define the range for the red color in HSV
    # Red color is split into two ranges in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Define the range for the blue color in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Define the range for the yellow color in HSV
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    # Create masks for the two ranges of red
    mask_red1 = cv2.inRange(imgHSV, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(imgHSV, lower_red2, upper_red2)

    # Create a mask for blue
    mask_blue = cv2.inRange(imgHSV, lower_blue, upper_blue)

    # Create a mask for yellow
    mask_yellow = cv2.inRange(imgHSV, lower_yellow, upper_yellow)

    # Combine the red, blue, and yellow masks
    mask_red_and_blue_yellow = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red_and_blue_yellow = cv2.bitwise_or(mask_red_and_blue_yellow, mask_blue)
    mask_red_and_blue_yellow = cv2.bitwise_or(mask_red_and_blue_yellow, mask_yellow)

    # The final result: white for red, blue, and yellow regions, black for others
    result = cv2.bitwise_and(imgBGR, imgBGR, mask=mask_red_and_blue_yellow)

    # Convert result to binary (white for red, blue, yellow regions, black for everything else)
    result_binary = cv2.inRange(result, (1, 1, 1), (255, 255, 255))  # All non-zero becomes white

    return result_binary

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=1.33):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects


    for contour in contours:
        area = cv2.contourArea(contour)

        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        if area >= 10000 or len(approx) <= 8:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


# Kamera-Stream
cap = cv2.VideoCapture(0)
cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_bin = preprocess(frame)
    min_area = img_bin.shape[0] * frame.shape[1] / (25 * 25)
    # Shape Detection
    rects = contour_detect(img_bin, min_area=min_area)   # get x,y,h and w.
    img_bbx = frame.copy()
    for rect in rects:

        # rect[2] is width and rect[3] for height
        if rect[2] > 100 and rect[3] > 100:             #only detect those signs whose height and width >100


            crop_img = frame[rect[1]:rect[1]+rect[2], rect[0]:rect[0]+rect[3]]
            crop_img = cv2.resize(crop_img, (32,32)) / 255
            crop_img = np.reshape(crop_img, (1,32,32,3))# Resize

            predictions = model.predict(crop_img)
            label = labels[predictions.argmax()]

            predictions = model.predict(crop_img)
            label = labels[predictions.argmax()]
            if np.amax(predictions):
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                #write class name on the output screen
                cv2.putText(frame, label, (rect[0], rect[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)






    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

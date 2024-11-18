import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

model = load_model('traffic_sign_model.keras')
labels = ["20", "30", "50", "60", "70", "80",
          "80 aufgehoben", "100", "120", "Überholen verboten", "Lkw überholen verboten",
          "Vorfahrt", "Vorfahrtsstraße", "Vorfahrt achten", "Stop",
          "Durchfahrt verboten", "LKW verboten", "Durchfahrt verboten Einbahnstraße",
          "Achtung", "Scharfe Linkskurve", "Scharfe Rechtskurve", "Mehrere Kurven", "Bodenwelle", "Achtung Eis",
          "Achtung Verengung", "Achtung Baustelle", "Achtung Ampel", "Achtung Fußgänger", "Achtung Kinder", "Achtung Fahrrad",
          "Achtung Frost", "Achtung Wildwechsel", "Unbegrenzt!!!", "Nur rechts fahren", "Nur links fahren",
          "Gerade ausfahren", "Gerade aus und rechts fahren", "Gerade aus und links", "Rechts vorbeifahren", "Links vorbeifahren",
          "Kreisverkehr", "Überholverbot Ende", "LKW Verbot beenden"]  # Klassenlabels

def preprocess_img(imgBGR, erode_dilate=True):  # pre-processing fro detect signs in  image.
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
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

    img_bin = preprocess_img(frame, False)
    min_area = img_bin.shape[0] * frame.shape[1] / (25 * 25)
    # Shape Detection
    rects = contour_detect(img_bin, min_area=min_area)   # get x,y,h and w.
    img_bbx = frame.copy()
    for rect in rects:
        print("yes")
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

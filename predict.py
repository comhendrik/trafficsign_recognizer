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



# Kamera-Stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Bildvorverarbeitung
    img = cv2.resize(frame, (32,32)) / 255
    img = np.reshape(img, (1,32,32,3))# Resize

    # Vorhersage
    predictions = model.predict(img)
    label = labels[predictions.argmax()]

    # Ergebnis anzeigen
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

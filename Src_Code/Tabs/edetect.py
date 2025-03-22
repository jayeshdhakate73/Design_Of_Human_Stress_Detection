import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import csv
import pandas as pd
import os

# Load the Pre-trained Emotion Detection Model
model = tf.keras.models.load_model('Src_Code/model.keras')
st.write("Model loaded successfully.")

# Emotion Mapping
detection_labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
stress_mapping = {"Angry": 2, "Fearful": 2, "Sad": 2, "Disgusted": 1, "Surprised": 1, "Happy": 0, "Neutral": 0}

# Load Face Detection Model
face_net = cv2.dnn.readNetFromCaffe("Src_Code/deploy.prototxt.txt", "Src_Code/res10_300x300_ssd_iter_140000.caffemodel")

def detect_stress():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam")
        return
    
    record_duration = 10  # seconds
    start_time = time.time()
    results = []
    
    stframe = st.empty()
    while time.time() - start_time < record_duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY, endX, endY = max(0, startX), max(0, startY), min(w - 1, endX), min(h - 1, endY)

            face_roi = frame[startY:endY, startX:endX]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized.astype("float32") / 255.0
            roi_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)

            prediction = model.predict(roi_input)
            predicted_index = int(np.argmax(prediction))
            predicted_emotion = detection_labels[predicted_index]
            stress_level = stress_mapping[predicted_emotion]

            current_time = time.time() - start_time
            results.append((round(current_time, 2), predicted_emotion, stress_level))
            
            # Display Camera Output
            cv2.putText(frame, predicted_emotion, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        stframe.image(frame, channels="BGR")
    
    cap.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    output_file = "stress_results.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Emotion", "Stress Level"])
        writer.writerows(results)
    
    # Determine final stress result
    df = pd.read_csv(output_file)
    df['Stress Level'] = pd.to_numeric(df['Stress Level'], errors='coerce')
    avg_stress = df['Stress Level'].mean()
    high_stress_count = (df['Stress Level'] == 2).sum()
    total_frames = len(df)
    high_stress_percentage = (high_stress_count / total_frames) * 100
    
    is_stressed = avg_stress >= 1.0 or high_stress_percentage >= 40
    
    if is_stressed:
        st.error("⚠️ You appear to be stressed!")
    else:
        st.success("✅ You are not stressed.")

def app():
    st.title("Live Emotion Detection")
    if st.button("Start Camera"):
        detect_stress()

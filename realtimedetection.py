# from keras.models import model_from_json
# import cv2
# import numpy as np

# json_file = open("signlanguagedetectionmodel48x48.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("signlanguagedetectionmodel48x48.h5")

# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1,48,48,1)
#     return feature/255.0

# cap = cv2.VideoCapture(0)
# label = ['A', 'M', 'N', 'S', 'T', 'blank']
# while True:
#     _,frame = cap.read()
#     cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
#     cropframe=frame[40:300,0:300]
#     cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
#     cropframe = cv2.resize(cropframe,(48,48))
#     cropframe = extract_features(cropframe)
#     pred = model.predict(cropframe) 
#     prediction_label = label[pred.argmax()]
#     cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
#     if prediction_label == 'blank':
#         cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
#     else:
#         accu = "{:.2f}".format(np.max(pred)*100)
#         cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
#     cv2.imshow("output",frame)
#     cv2.waitKey(27)
    
# cap.release()
# cv2.destroyAllWindows()


import streamlit as st
from keras.models import model_from_json
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained Keras model
@st.cache_resource
def load_model():
    json_file = open("signlanguagedetectionmodel48x48.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("signlanguagedetectionmodel48x48.h5")
    return model

# Extract features function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Predict sign language label
def predict_sign_language(frame, model):
    label = ['A', 'B', 'C', 'D', 'E', 'F' , 'G' , 'H' , 'I' , 'J' , 'K' , 'L' , 'M' , 'N' , 'O' , 'P' , 'Q' , 'R' , 'S' , 'T' , 'U' , 'V' , 'W' , 'X' , 'Y' , 'Z' 'blank']
    crop_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (48, 48))
    crop_frame = extract_features(crop_frame)
    
    pred = model.predict(crop_frame)
    prediction_label = label[pred.argmax()]
    confidence = np.max(pred) * 100
    return prediction_label, confidence

# Streamlit Web App
st.title("SignBridge India")

# Load Keras Model
model = load_model()

# Webcam Video Capture and Streamlit Display
st.text("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)
label_placeholder = st.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to access the webcam")
        break

    # Draw the region for detection
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]

    # Predict the sign language label
    prediction_label, confidence = predict_sign_language(crop_frame, model)

    # Display prediction and confidence
    label_placeholder.text(f"Prediction: {prediction_label} with {confidence:.2f}% confidence")

    # Display the frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    cap.release()
    st.write("Webcam stopped")
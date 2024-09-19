import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Directory setup
directory = 'SignImage48x48/'
if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(f'{directory}/blank'):
    os.mkdir(f'{directory}/blank')
for i in range(65, 91):
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')

# Define key to directory mapping
key_to_dir = {
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H',
    'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P',
    'q': 'Q', 'r': 'R', 's': 'S', 't': 'T', 'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X',
    'y': 'Y', 'z': 'Z', '.': 'blank'
}

# Function to process frame and save image
def process_frame(frame, key, count):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48))
    filename = os.path.join(directory, key_to_dir[key], f'{count}.jpg')
    cv2.imwrite(filename, frame)

# Streamlit app
st.title("Sign Language Detection Data Collection")

# Camera input
video_file = st.camera_input("Capture images")

if video_file:
    # Convert video file to numpy array
    video_bytes = video_file.read()
    video_array = np.frombuffer(video_bytes, np.uint8)
    video = cv2.imdecode(video_array, cv2.IMREAD_COLOR)

    if video is not None:
        st.image(video, channels="BGR", caption="Live Video Feed")
        
        # Initialize frame count
        frame_count = {key: 0 for key in key_to_dir.keys()}

        # Display buttons for each category
        selected_category = st.radio("Select Category", list(key_to_dir.keys()))
        if st.button("Capture and Save Image"):
            # Save the current frame
            process_frame(video, selected_category, frame_count[selected_category])
            frame_count[selected_category] += 1
            st.success(f"Image saved in category {selected_category.upper()}")
else:
    st.write("Please use the camera input to capture an image.")
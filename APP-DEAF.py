import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from ultralytics import YOLO
from PIL import Image

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
model=YOLO('yolovm.pt')
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.result_text = ""
        self.Labels =[
            'للغة العربية -arabic lan','- - -signlan','-raa', '-shen-','-taa','-thaa','-zaa',' Eight','Ingoodhealth',' Melon','Photographer',' afraid',
            'alaph-', 'ayn-',' ba-','- baba',' book-',' caff-',' company','- daa-','- dad-','- dal-',' day',
            'dhal-',' eat','faa', 'family','friday',' friend_', 'gen-','grandfather',' ha-', "ha---", 'haa-', 'home_house', 'jem-', 'khaa-', 'lam-', 'learn -', 'mall', 'mama', 'meem-','monday', 'month','mosque'
            ,'noon-', 'playground', 'qaf-','sad-', 'salam- -','saturday', 'school_', 'sick', 'sin-', 'sleep', 'sorry', 'sunday', 'ta-', 'thursday', 'tuesday-', 'university','wau-','yaa','zero'
            ]

        self.model = YOLO('C:/Users/Lenovo/Desktop/local omdena -jordan/code/task5-test/best.pt')
        self.LabelArabic=[]
        


        self.output_filename = 'output.mp4'
        self.output_codec = 'XVID'
        self.output_fps = 30.0
        self.output = None

    def start_output_writer(self, frame):
        # Get the frame dimensions
        height, width, _ = frame.shape

        # Create a VideoWriter object to write the frames with detections to the output video file
        self.output = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*self.output_codec), self.output_fps, (width, height))
    def stop_output_writer(self):
        # Release the VideoWriter object
        self.output.release()
    def transform(self, frame):
        # Process individual frames from real-time video stream
        if frame is None or len(frame) == 0:
                    return None
      
        frame= np.array(frame)

        # Check if the frame array is empty
     
        # Resize the frame to the input size expected by the model
        resized_frame = cv2.resize(frame, (640, 640))

        # Pass the frame through the model
        detections = self.model.predict(resized_frame,conf=.5)
        # Process the detection results (bounding boxes, class labels, etc.)
        # Example: Extract the bounding boxes and class labels from the detections tensor
        boxes = detections[0].boxes
        labels= boxes.cls
        print(boxes)
        print(labels)
        class_label=""
        # Visualize the predictions by drawing bounding boxes and labels on the frame
        if boxes is not None and labels is not None:
            for box, label in zip(boxes.xyxy, labels):
                x1, y1, x2, y2 = box.int().tolist()
                class_label = self.Labels[int(labels[-1])] 
                #text=self.LabelArabic[int(labels[-1])] 
                # Map the label index to the corresponding class label
                print(class_label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the processed frame
                st.image(frame, channels="BGR")

        # Extract text from the processed frame
        result =str(class_label)
        self.result_text = result
        st.write("**class EN:** ")
        st.write(self.result_text)
        st.write("**Word Arabic :**")
       # st.write(text)

        frame= frame.astype(np.uint8)
         # Write the frame with detections to the output video file
        if self.output is True:
            self.start_output_writer(frame)

            self.output.write(frame)

        return frame

def main():
    st.title("Deaf Assistance App")
    st.sidebar.header("Input Options")
    input_option = st.sidebar.selectbox(
        "Select Input Option",
        ("Video from Device", "Image from Device", "Real-time Video Stream")
    )



    if input_option == "Video from Device":
        
        video_file = st.file_uploader("Upload video file", type=["mp4", "avi"])
        if video_file:
                               # Process video file
            st.write("Processing video from device:", video_file.name)
            #output = av.open('output.mp4', mode='rw')
          
            video_file = open(video_file.name, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            input_container = av.open(video_file.name)

            input_stream = input_container.streams.video[0]
            input_width = input_stream.width
            input_height = input_stream.height
                # Create an output container to store the transformed frames
           

            # Create an instance of the VideoTransformer
            video_transformer = VideoTransformer()
            video_transformer.output = True
            # Process each frame in the input video
            for frame in input_container.decode(video=0):
                image = frame.to_ndarray(format='bgr24')

                # Apply the transformation using the VideoTransformer
                transformed_frame = video_transformer.transform(image)

                # Convert the transformed frame to the av.VideoFrame format
# Convert the transformed frame to the av.VideoFrame format
            st.text("**detction Output video**")
            video_file = open(video_transformer.output_filename, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
                


# Display the processed video
        



    elif input_option == "Image from Device":
        image_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"])
        if image_file:
            # Process image file
            st.write("Processing image from device:", image_file.name)
            image = Image.open(image_file.name)
            st.image(image)
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            video_transformer = VideoTransformer()
            transformed_image = video_transformer.transform(opencv_image)
            st.image(transformed_image )


          



    elif input_option == "Real-time Video Stream":
            st.write("Real-time video stream")
            video_transformer = VideoTransformer()

            webrtc_ctx = webrtc_streamer(
                key="example",
                rtc_configuration=RTC_CONFIGURATION,
                video_transformer_factory=VideoTransformer,
            )

            if webrtc_ctx.video_transformer:
                stream = webrtc_ctx.video_transformer.transformed_video_stream
                st.video(stream)

                # Start the output video writer
                video_transformer.start_output_writer(frame)

                while True:
                    transformed_frame = webrtc_ctx.video_transformer.transform(frame)
                    st.write("Processed video stream")
                    st.video(transformed_frame)

                    # Write the transformed frame to the output video file
                    video_transformer.output.write(transformed_frame)

                # Stop the output video writer
                video_transformer.stop_output_writer()

if __name__ == "__main__":
    main()

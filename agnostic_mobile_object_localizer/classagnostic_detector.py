#https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1 : A class-agnostic object detection framework . Network built on MobileNetv2 backbone with SSDLite detection head

import cv2
import numpy as np
import tensorflow as tf
from skimage import io
from PIL import Image
import requests
import matplotlib.pyplot as plt

np.random.seed(0)


colors = np.random.randint(255, size=(100, 3), dtype=int)

class GenericDetector():

    def __init__(self, model_path, threshold = 0.2):

        self.threshold = threshold

        # Initialize model
        self.initialize_model(model_path)
        

    def __call__(self, image):

        return self.detect_objects(image)

    def initialize_model(self, model_path):

        self.model = tf.saved_model.load(model_path).signatures['default']
        

        # Get model info
        self.getModel_input_details()

    def detect_objects(self, image):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        output = self.inference(input_tensor)
        print(output)

        # # Process output data
        detections = self.process_output(output)

        return detections

    def prepare_input(self, image):

        
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        img = image[..., ::-1]
    

        input_tensor = cv2.resize(img, (self.input_width,self.input_height))
        input_tensor = input_tensor[np.newaxis,:,:,:]      

        return tf.convert_to_tensor(input_tensor.astype(np.float32))

    def inference(self, input_tensor):

        # Peform inference
        return self.model(input_tensor)

    def process_output(self, output):  

        # Get all output details
        boxes = output['detection_boxes'].numpy()[0]
        classes = output['detection_classes'].numpy()[0]
        scores = output['detection_scores'].numpy()[0]
        num_objects = int(output['num_detections'][0])

        results = []
        for i in range(num_objects):
            if scores[i] >= self.threshold:
                result = {
                  'bounding_box': boxes[i],
                  'class_id': classes[i],
                  'score': scores[i]
                }
                results.append(result)
        return results

    def getModel_input_details(self):

        input_shape =  self.model.inputs[0].shape
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

    @staticmethod
    def draw_detections(image, detections):
        
        image = np.array(image)
        img_height, img_width, _ = image.shape

        for idx, detection in enumerate(detections):
            box = detection['bounding_box']
            label = detection['class_id']
            score = detection['score']
            y1 = (img_height * box[0]).astype(int)
            y2 = (img_height * box[2]).astype(int)
            x1 = (img_width * box[1]).astype(int)
            x2 = (img_width * box[3]).astype(int)

            cv2.rectangle(image, (x1, y1), (x2, y2), (int(colors[idx,0]), int(colors[idx,1]), int(colors[idx,2])), 5)
            cv2.putText(image, "Entity", (x1, y1+10),cv2.FONT_HERSHEY_SIMPLEX,5,(int(colors[idx,0]), int(colors[idx,1]), int(colors[idx,2])),15)

        return image


if __name__ == '__main__':

    model_path="models/saved_model"
    threshold = 0.2

    # Initialize object detection model
    detector = GenericDetector(model_path, threshold)

    # Read RGB image

    image = Image.open("/home/adv8/Documents/HiWi/20220715_084700793_iOS.jpg")

    # Draw the detected objects
    detections = detector(image)
    detection_img = detector.draw_detections(image, detections)
        
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.imshow("Detections", detection_img)
    cv2.waitKey(0)

from styx_msgs.msg import TrafficLight

import sys
import tarfile
import os

import tensorflow as tf
import keras
import numpy as np
import cv2


class TLClassifier(object):
    def __init__(self, dataset, traffic_light_class_id):
        #TODO load classifier
        self.dataset = dataset
        self.dataset_url = 'http://download.tensorflow.org/models/object_detection/' + dataset + '.tar.gz'
        model_dir = keras.utils.get_file(
            fname=dataset, 
            origin=self.dataset_url,
            untar=True)
        model_path = os.path.join(model_dir, "frozen_inference_graph.pb")
        
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.detection_graph = detection_graph
        self.session = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        #model_path2 = os.path.join(model_dir, "saved_model")
        #os.rename(str(model_path), str(model_path2))
        #self.model = tf.keras.models.load_model(str(model_path2))
        #self.model = tf.saved_model.load(str(model_path2))
        self.class_id = traffic_light_class_id

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        light_exist = False
        light_prediction = None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (im_width, im_height, _) = image_rgb.shape
        image_np = np.expand_dims(image_rgb, axis=0)

        # Actual detection.
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.session.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np})
        
        sq_scores = np.squeeze(scores)
        sq_classes = np.squeeze(classes)
        sq_boxes = np.squeeze(boxes)

        sel_id = np.logical_and(sq_classes == self.class_id, sq_scores >0.2)
        sq_boxes = sq_boxes[sel_id]
        if len(sq_boxes) > 0:
            image = self.crop_img(image, sq_boxes[0])
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            green_mask = cv2.inRange(image_hsv, (35, 0, 0), (70, 255,255))
            green = cv2.bitwise_and(image, image, mask=green_mask)
            
            red_mask_down = cv2.inRange(image_hsv, (0,50,20), (10,255,255))
            red_mask_up = cv2.inRange(image_hsv, (175,50,20), (185,255,255))
            
            red_mask = cv2.bitwise_or(red_mask_down, red_mask_up)
            red = cv2.bitwise_and(image, image, mask=red_mask)
            
            red = np.sum(red)
            green = np.sum(green)
            
            if (red <= green and green > image.size*0.2):
                return TrafficLight.GREEN
            else:
                return TrafficLight.RED
        return TrafficLight.UNKNOWN
    
    def crop_img(self, image, box):
        h, w, d = image.shape
        bt, l, tp, r = box
        bt = int(bt*h)
        l = int(l*w)
        tp = int(tp*h)
        r = int(r*w)
        return image[bt:tp, l:r, :]

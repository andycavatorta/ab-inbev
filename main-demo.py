"""
todo:

old:
    create new images for feedback

new features:
    detect / fix overlap
    email report
        slick formatting
        inventory
        map
        strangers
        changes / patterns



"""
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import commands
import cv2
import datetime
import json
import numpy as np
import os
from os import environ
from os import walk
from os.path import join, dirname
import RPi.GPIO as GPIO
import smtplib
import shutil
import tensorflow as tf
import time
#import zipfile

PATH_FOR_THIS_FILE = os.path.dirname(os.path.realpath(__file__))

class Camera():
        def __init__(self, images_folder, cam_id, pin, x_offset, y_offset):
            self.images_folder = images_folder
            self.cam_id = cam_id
            self.pin = pin
            self.x_offset = x_offset
            self.y_offset = y_offset
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        def take_photo(self):
            print  'Camera {} taking picture'.format(self.cam_id)
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(1)
            #filename = '{}/image_{}.png'.format(self.images_folder,self.cam_id)
            filename = '%s/image_%s.png' % (self.images_folder,self.cam_id)
            try: 
                cap = cv2.VideoCapture(0)
                cap.set(3,1280)
                cap.set(4,720)
                if not cap.isOpened():
                    while not cap.isOpened():
                        print "Camera capture not open ... trying to fix it..."
                        cap.open()
                        time.sleep(1)
                else:
                    print filename
                    ret, frame = cap.read()
                    cv2.imwrite(filename,frame)
                    cap.release()
                    print "Picture taken"
            except Exception as e:
                  print "Oops! something went wrong %s" % (e)
            finally:
                 GPIO.output(self.pin, GPIO.LOW)
            time.sleep(1)
            return [filename, self.x_offset, self.y_offset]

class Cameras():
        def __init__(self):
            GPIO.setmode(GPIO.BCM)
            self.pins = [2,3,4,14,15,17,18,27,22,23,24,10]
            self.x_offsets = [0,800,1600,0,800,1600,0,800,1600,0,800,1600]
            self.y_offsets = [0,0,0,450,450,450,900,900,900,1350,1350,1350]
            #self.x_offsets = [1600,800,0,1600,800,0,1600,800,0,1600,800,0,]
            #self.y_offsets = [1350,1350,1350, 900,900,900,450,450,450,0,0,0  ]
            self.images_folder_name = ("%s/camera_capture_images") % (os.path.dirname(os.path.realpath(__file__)))
            #os.makedirs(self.images_folder_name)
            self.cameras = [Camera(self.images_folder_name, c, self.pins[c], self.x_offsets[c], self.y_offsets[c]) for c in range(12)]
            self.lastImages = []
        def take_all_photos(self):
            self.empty_directory()
            self.set_all_pins_low() # just in case
            for cam in self.cameras:
                metadata = cam.take_photo()
                self.lastImages.append(metadata)
        def set_all_pins_low(self):
            for pin in self.pins:
                GPIO.output(pin, GPIO.LOW)
        def get_images_folder(self):
            return self.images_folder_name
        def get_capture_data(self):
            return self.lastImages
        def get_offset_from_id(self, id):
            return [self.x_offsets[id],self.y_offsets[id]]
        def empty_directory(self):
            for file in os.listdir(self.images_folder_name):
                file_path = os.path.join(self.images_folder_name, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)

class ImageParser(): # class not necessary.  used for organization
    def __init__(self):
        self.parsedCaptures = [] # 2D list of capture:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.foldername = ("%s/cropped") %(dir_path)
        #os.makedirs(self.foldername)
    def empty_directory(self):
        for file in os.listdir(self.foldername):
            file_path = os.path.join(self.foldername, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def get_foldername(self):
        return self.foldername

    def get_parsed_images(self):
        return self.parsedCaptures

    def undistort_image(self, image):
        width = image.shape[1]
        height = image.shape[0]
        distCoeff = np.zeros((4,1),np.float64)
        k1 = -6.0e-5; # negative to remove barrel distortion
        k2 = 0.0;
        p1 = 0.0;
        p2 = 0.0;
        distCoeff[0,0] = k1;
        distCoeff[1,0] = k2;
        distCoeff[2,0] = p1;
        distCoeff[3,0] = p2;
        # assume unit matrix for camera
        cam = np.eye(3,dtype=np.float32)
        cam[0,2] = width/2.0  # define center x
        cam[1,2] = height/2.0 # define center y
        cam[0,0] = 10.        # define focal length x
        cam[1,1] = 10.        # define focal length y
        # here the undistortion will be computed
        return cv2.undistort(image,cam,distCoeff)

    def process_image(self, filepath, camera_id, offset_x, offset_y):
        print "Processing image...", camera_id, filepath
        parsedImageMetadata = [] 
        self.parsedCaptures.append(parsedImageMetadata)# images are introduce in order of cap_id, so list index == cap_id
        img_for_cropping = cv2.imread(filepath) # read image into memory
        img_for_cropping = cv2.resize(img_for_cropping, (800,450), cv2.INTER_AREA) # resize image
        img_for_cropping = self.undistort_image(img_for_cropping) # get unbent!

        img_for_circle_detection = cv2.imread(filepath,0) # read image into memory
        img_for_circle_detection = cv2.resize(img_for_circle_detection, (800,450), cv2.INTER_AREA) # resize image
        img_for_circle_detection = self.undistort_image(img_for_circle_detection) # get unbent!
        # cv2.imshow('dst', img_for_circle_detection)
        height, width = img_for_circle_detection.shape

        img_for_circle_detection = cv2.medianBlur(img_for_circle_detection,21)

        #testFileName = "{}_1_median.png".format(camera_id)
        #cv2.imwrite(testFileName ,img_for_circle_detection)

        img_for_circle_detection = cv2.blur(img_for_circle_detection,(1,1))

        #testFileName = "{}_2_blur.png".format(camera_id)
        #cv2.imwrite(testFileName ,img_for_circle_detection)

        img_for_circle_detection = cv2.Canny(img_for_circle_detection, 0, 23, True)

        #testFileName = "{}_3_canny.png".format(camera_id)
        #cv2.imwrite(testFileName ,img_for_circle_detection)


        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.1

        params.filterByArea = True
        params.minArea = 100
        #params.maxCircularity = 0

        # Read image
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(im)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show keypoints

        testFileName = "{}_2_blob.png".format(camera_id)
        cv2.imwrite(testFileName ,im_with_keypoints)


        #img_for_circle_detection = cv2.medianBlur(img_for_circle_detection,21)
        #img_for_circle_detection = cv2.blur(img_for_circle_detection,(1,1))
        #img_for_circle_detection = cv2.Canny(img_for_circle_detection, 0, 23, True)
        img_for_circle_detection = cv2.adaptiveThreshold(img_for_circle_detection,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)


        print "Detecting circles..."
        circles = cv2.HoughCircles(img_for_circle_detection,cv2.HOUGH_GRADIENT,1,150, param1=70,param2=28,minRadius=30,maxRadius=80)

        testFileName = "{}_4_circleDetectionTest.png".format(camera_id)
        cv2.imwrite(testFileName ,img_for_circle_detection)
        testFileName = "{}_0_croppingTest.png".format(camera_id)
        cv2.imwrite(testFileName ,img_for_cropping) 

        circles = np.uint16(np.around(circles))
        margin = 30
        for x, y, radius in circles[0,:]:
            x=int(x)
            y=int(y)
            radius=int(radius)
            #leftEdge = x-radius-margin
            #rightEdge = x+radius+margin
            #topEdge = y-radius-margin
            #bottomEdge = y+radius+margin
            #if leftEdge < 0 or  rightEdge > width or topEdge < 0 or bottomEdge > height:
            #   continue
            leftEdge = x-radius-margin if x-radius-margin >= 0 else 0
            rightEdge = x+radius+margin if x+radius+margin <= width else width
            topEdge = y-radius-margin if y-radius-margin >=0 else 0
            bottomEdge = y+radius+margin if y+radius+margin <= height else height
            crop_img = img_for_cropping[topEdge:bottomEdge, leftEdge:rightEdge]
            imageName = 'image_%s_%s_%s.jpg'%(camera_id,x, y)
            pathName = '%s/%s'%(self.foldername, imageName)
            cv2.imwrite(pathName,crop_img)
            # draw the outer circle
            cv2.circle(img_for_cropping,(x,y),radius,(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img_for_cropping,(x,y),2,(0,0,255),3)
            #print len(circles)
            totalX = x + offset_x
            totalY = y + offset_y
            parsedImageMetadata.append( {
                'capture':camera_id,
                'imageName':imageName,
                'pathName':pathName,
                'x':x,
                'y':y,
                'totalX':totalX,
                'totalY':totalY,
                'radius':radius,
                'leftEdge':leftEdge,
                'rightEdge':rightEdge,
                'topEdge':topEdge,
                'bottomEdge':bottomEdge,
                'label':"",
                'confidence':0
            } )
            #print "detected circle:", repr(x), repr(y), repr(radius), leftEdge, rightEdge, topEdge, bottomEdge
        # cv2.imshow('detected circles',img_for_cropping)
        cv2.destroyAllWindows()
        #print parsedImageMetadata
        print "Processing image done"

    def processImages(self, captureLIst):
        self.empty_directory()
        for index, cap_metadata in enumerate(captureLIst):
            self.process_image(cap_metadata[0],index, cap_metadata[1], cap_metadata[2])


class Classifier():
    def __init__(self):
        self.cooling_period = 10
        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line 
            in tf.gfile.GFile("image_classifier/tf_files/retrained_labels.txt")]

    def check_temp(self):
        print "temperature=", commands.getstatusoutput("/opt/vc/bin/vcgencmd measure_temp")
    
    def classify_images(self, imageMetadataList):
        with tf.gfile.FastGFile("image_classifier/tf_files/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            image_count = 0
            for camera in  imageMetadataList:
                for imageMetadata in camera:
                    print "classifying image", image_count
                    image_count += 1
                    image_data = tf.gfile.FastGFile(imageMetadata["pathName"], 'rb').read()
                    
                    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')# Feed the image_data as input to the graph and get first prediction
                    predictions = sess.run(softmax_tensor, \
                             {'DecodeJpeg/contents:0': image_data})
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]# Sort to show labels of first prediction in order of confidence
                    #print "top_k=", repr(top_k)
                    self.check_temp()
                    time.sleep(2)
                    for node_id in top_k:
                        human_string = self.label_lines[node_id]
                        score = predictions[0][node_id]
                        #print('%s (score = %.5f)' % (human_string, score))
                    # print(self.label_lines[top_k[0]])
                    imageMetadata["label"] = self.label_lines[top_k[0]]
                    imageMetadata["confidence"] = predictions[0][top_k[0]]


def data_viz(img_metadata):
    canvas = np.zeros((2400,2400,3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for camera in img_metadata:
        for imageMetadata in camera:
            cv2.circle(canvas, (imageMetadata['totalX'],imageMetadata['totalY']),40, (200,200,200), -1)
            cv2.putText(canvas, "%s - %s" % (imageMetadata['label'],imageMetadata['capture']), (imageMetadata['totalX']-30,imageMetadata['totalY']+50), font, 0.5,(100,100,100),2,cv2.LINE_AA)
    cv2.imwrite('results.png',canvas)
    cv2.destroyAllWindows()


def print_temp():
    print "temperature=", commands.getstatusoutput("/opt/vc/bin/vcgencmd measure_temp")


class ProcessInventory():
    def __init__(self):
        self.confidence_threshold = 0.5
        self.data_raw = None
        self.data_processed = None
        self.inventory_template = {
            "budlight":0,
            "budweiser":0,
            "corona":0,
            "hoegaarden":0,
            "platinum":0,
            "stella":0,
            "ultra":0
        }
    def process_inventory_data(self, data):
        self.data_raw = data
        data_filtered = self.filter_low_confidence(self.data_raw)
        self.data_processed =self.detect_overlaps(data_filtered)
        return self.data_processed
        #return self.collate_inventory()
    def filter_low_confidence(self, data):
        data_new = []
        for cam in self.data_raw:
            cam_new = []
            data_new.append(cam_new)
            for product in cam:
                if product["confidence"] >= self.confidence_threshold:
                    cam_new.append(product)
        return data_new
    def detect_overlaps(self, data):
        return data
    def collate_inventory(self):
        inventory = dict(self.inventory_template)
        for cam in self.data_processed:
            for product in cam:
                productName = product["label"]
                if productName in inventory.keys():
                    inventory[productName] += 1
                else:
                    print "ProcessInventory.collate_inventory: product name not found:", repr(product)
        return inventory


class Report():
    def __init__(self):
        self.to_field = "andycavatorta@gmail.com"
        self.from_field = "simurghnodes@gmail.com"
        self.password_field = "5ed0n6rum15"
        self.SMTP_server = "smtp.gmail.com"
        self.SMTP_port = 587

    def collect_inventory_data(self):
        #PATH_FOR_THIS_FILE
        d = {
            'cooler_id':0,
            'location':"Dark Matter",
            'address':"33 Flatbush Avenue, Brooklyn NY 11217 USA",
            'date_formatted':time.strftime('%A, %B %d %Y at %H:%M:%S'),
            'inventory':[], # tabular data
            'strangers':[], # image paths
            'mapImagePath':""
        }

    def generate_email(self):  
        pass 

    def send_email(self, inventory):
        msg = "asdf"
        server = smtplib.SMTP(self.SMTP_server, self.SMTP_port)
        server.starttls()
        server.login(self.from_field, self.password_field)
        server.sendmail(self.from_field, self.to_field, msg)
        server.quit()


def main():
    # create instances
    print_temp()
    cameras = Cameras()
    imageparser = ImageParser()
    classifier = Classifier()
    processinventory = ProcessInventory()
    report = Report()
    #while True:
    print "starting main inventory loop"
    print_temp()
    cameras.take_all_photos()
    print_temp()
    time.sleep(1)
    capture_list = cameras.get_capture_data()
    print_temp()
    imageparser.processImages(capture_list)
    print_temp()
    parsed_images = imageparser.get_parsed_images()
    parsed_folder_name = imageparser.get_foldername()
    print_temp()
    classifier.classify_images(parsed_images)
    print_temp()
    print parsed_images
    parsed_images_processed = processinventory.process_inventory_data(parsed_images)
    print_temp()
    inventory = processinventory.collate_inventory()
    print "inventory=", repr(inventory)
    print parsed_images_processed
    print_temp()
    data_viz(parsed_images_processed)
    print_temp()
    report.send_email(inventory)
    time.sleep(60)
    print_temp()
main()

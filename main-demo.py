import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np
import os
import zipfile
import json
from os.path import join, dirname
from os import environ
#from watson_developer_cloud import VisualRecognitionV3
import time
import datetime
import RPi.GPIO as GPIO

from classifier import Classifier
from os import walk
image_classifier = Classifier()



dir_path = os.path.dirname(os.path.realpath(__file__))
now = datetime.datetime.now()
realnow = now.strftime("%Y-%m-%d-%H-%M-%S")
foldername = ("%s/cropped/%s") %(dir_path, realnow)
caps_positions = []
results_json = []
#images_folder = "%s" % (realnow)
#os.makedirs(images_folder)
#os.makedirs(foldername)



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
            self.x_offsets = [0,0,0,0,0,0,0,0,0,0,0,0]
            self.y_offsets = [0,0,0,0,0,0,0,0,0,0,0,0]
            self.images_folder_name = ("%s/camera_capture_images/%s") % (os.path.dirname(os.path.realpath(__file__)), now.strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(self.images_folder_name)
            self.cameras = [Camera(self.images_folder_name, c, self.pins[c], self.x_offsets[c], self.y_offsets[c]) for c in range(12)]
            self.lastImages = []
        def take_all_photos(self):
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
 

class ImageParser(): # class not necessary.  used for organization
    def __init__(self):
        self.parsedCaptures = [] # 2D list of capture:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        now = datetime.datetime.now()
        realnow = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.foldername = ("%s/cropped/%s") %(dir_path, realnow)
        os.makedirs(self.foldername)

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

    def process_image(self, filepath, camera_id):
        print "Processing image...", camera_id, filepath
        parsedImageMetadata = [] 
        self.parsedCaptures.append(parsedImageMetadata)# images are introduce in order of cap_id, so list index == cap_id
        img_for_cropping = cv2.imread(filepath)
        img_for_cropping = cv2.resize(img_for_cropping, (800,450), cv2.INTER_AREA)
        img_for_cropping = self.undistort_image(img_for_cropping)

        img_for_circle_detection = cv2.imread(filepath,0)
        img_for_circle_detection = cv2.resize(img_for_circle_detection, (800,450), cv2.INTER_AREA)
        img_for_circle_detection = self.undistort_image(img_for_circle_detection)
        # cv2.imshow('dst', img_for_circle_detection)
        height, width = img_for_circle_detection.shape
        img_for_circle_detection = cv2.medianBlur(img_for_circle_detection,21)
        img_for_circle_detection = cv2.blur(img_for_circle_detection,(1,1))
        img_for_circle_detection = cv2.Canny(img_for_circle_detection, 0, 23, True)
        img_for_circle_detection = cv2.adaptiveThreshold(img_for_circle_detection,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)

        print "Detecting circles..."
        circles = cv2.HoughCircles(img_for_circle_detection,cv2.HOUGH_GRADIENT,1,150, param1=70,param2=28,minRadius=30,maxRadius=80)
        circles = np.uint16(np.around(circles))
        margin = 30
        for x, y, radius in circles[0,:]:

            leftEdge = x-radius-margin if int(x)-int(radius)-margin >= 0 else 0
            rightEdge = int(x)+int(radius)+margin if int(x)+int(radius)+margin <= width else width
            topEdge = int(y)-int(radius)-margin if int(y)-int(radius)-margin >=0 else 0
            bottomEdge = int(y)+int(radius)+margin if int(y)+int(radius)+margin <= height else height

            crop_img = img_for_cropping[topEdge:bottomEdge, leftEdge:rightEdge]

            imageName = 'image_%s_%s_%s.jpg'%(camera_id,x, y)
            pathName = '%s/%s'%(self.foldername, imageName)
            cv2.imwrite(pathName,crop_img)
            # draw the outer circle
            cv2.circle(img_for_cropping,(x,y),radius,(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img_for_cropping,(x,y),2,(0,0,255),3)
            #print len(circles)
            parsedImageMetadata.append( {
                'capture':camera_id,
                'imageName':imageName,
                'pathName':pathName,
                'x':x,
                'y':y,
                'radius':radius,
                'leftEdge':leftEdge,
                'rightEdge':rightEdge,
                'topEdge':topEdge,
                'bottomEdge':bottomEdge
            } )
            print "detected circle:", repr(x), repr(y), repr(radius), leftEdge, rightEdge, topEdge, bottomEdge

        # cv2.imshow('detected circles',img_for_cropping)

        cv2.destroyAllWindows()
        print parsedImageMetadata
        print "Processing image done"

    def processImages(self, captureLIst):
        for index, cap_metadata in enumerate(captureLIst):
            self.process_image(cap_metadata[0],index)


cameras = Cameras()
imageparser = ImageParser()

cameras.take_all_photos()
time.sleep(1)

capture_list = cameras.get_capture_data()
imageparser = ImageParser()
imageparser.processImages(capture_list)





"""
cam = 1
for filename in os.listdir("%s/" % (images_folder)):
    if filename.endswith(".png"):
        filename = str(filename)
        print filename
        process_image(filename, cam)
        cam = cam + 1

run_tensorflow()
"""


##################################################################################################################


##############################
####### PROCESS IMAGE ########
##############################


##############################
###### SEND TO WATSON ########
##############################

def run_nn():
    #  michelle: 0ba27a1a79d9d2f600ad71cc3c32fada1499a3a2
    #  joao: 24f5aba0d5d54d4ecc619de28e71ddfca61c7559
    #  andy: e7b1ac2095bb25f8a919bb29c2c60af78701477c
    #  andy paid: 753a741d6f32d80e1935503b40a8a00f317e85c6 
    visual_recognition = VisualRecognitionV3('2016-05-20', api_key='753a741d6f32d80e1935503b40a8a00f317e85c6')

    "Uploading to the Neural Network..."
    # with open("%s.zip"%(foldername), 'rb') as image_file:
    results = []
    for root, dirs, filenames in os.walk(foldername):
        print filenames
        for file in filenames:
            with open(os.path.join(root, file), 'rb') as image_file:
                result = visual_recognition.classify(images_file=image_file,  classifier_ids=['beercaps_697951100'], threshold=0.99)
                results.append(result)

    global results_json
    results_json = results

    # with open('output.json', 'w') as file_:
    #     file_.write(results)

##############################
#### SEND TO TENSORFLOW ######
##############################
def run_tensorflow():
    guess = image_classifier.guess_image(foldername)
    print(guess)
    # f = []
    # for (dirpath, dirnames, filenames) in walk(foldername):
    #     f.extend(filenames)
    #     break
    # print("Found " + str(len(f)) + " files")
    # results = []
    # for image in f:
    #     guess = image_classifier.guess_image(foldername+'/'+image)
    #     results.append(guess)
    # print("done guessing")
    # print(results)
    # write_to_json(results)

##############################
######### DATA VIZ ###########
##############################

def data_viz(list_of_x, list_of_y, list_of_names, list_of_camera):

    img = np.zeros((1800,2400,3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x, y, name, camera in zip(list_of_x, list_of_y, list_of_names, list_of_camera):
        if camera == "2":
            x = int(x) + 800
        elif camera == "3":
            x = int(x) + (2*800)
        elif camera ==  "4":
            y = int(y) + 450
        elif camera == "5":
            x = int(x) + 800
            y = int(y) + 450
        elif camera == "6":
            x = int(x) + (2*800)
            y = int(y) + 450
        elif camera == "7":
            y = int(y) + (2*450)
        elif camera == "8":
            y = int(y) + (2*450)
            x = int(x) + 800
        elif camera == "9":
            y = int(y) + (2*450)
            x = int(x) + (2*800)
        elif camera == "10":
            y = int(y) + (3*450)
        elif camera == "11":
            y = int(y) + (3*450)
            x = int(x) + 800
        elif camera == "12":
            y = int(y) + (3*450)
            x = int(x) + (2*800)

        img = cv2.circle(img, (int(x),int(y)),40, (255,255,255), -1)
        cv2.putText(img, name, (int(x)-30,int(y)+60), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite('results.png',img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


##############################
####### PROCESS DATA #########
##############################

def process_data():

    list_of_x = []
    list_of_y = []
    list_of_names = []
    list_of_camera = []

    # with open('output.json') as json_data:
    #     d = json.load(json_data)

    d = results_json

    print d

    for files in d:   
        for images in files['images']:
            image_name = images['image'].rsplit('/',1)[-1]
            image_y = image_name.rsplit('_',1)[-1]
            image_y = image_y.rsplit('.',1)[-2]
            image_x = image_name.rsplit('_',2)[-2]
            camera = image_name.rsplit('_',3)[-3]
            list_of_x.append(str(image_x))
            list_of_y.append(str(image_y))
            list_of_camera.append(str(camera))

            for classy in images['classifiers']:
                for scores in classy['classes']:
                    list_of_names.append(str(scores['class']))

    budlight = 0
    stella = 0
    hoegaarden = 0
    budweiser = 0
    platinum = 0
    ultra = 0

    for name in list_of_names:
        if name == 'budlight':
            budlight=budlight+1
        elif name == 'stella':
            stella=stella+1
        elif name == 'hoegaarden':
            hoegaarden=hoegaarden+1
        elif name == 'budweiser':
            budweiser=budweiser+1
        elif name == 'platinum':
            platinum=platinum+1
        else:
            ultra=ultra+1

    print ("Budlights: %s | Stella: %s | Hoegaarden: %s | Budweiser: %s | Platinum: %s | Ultra: %s " % (budlight, stella, hoegaarden, budweiser, platinum, ultra))

    data_viz(list_of_x,list_of_y,list_of_names, list_of_camera)


##############################
######## NARRATIVE ###########
##############################






import cv2
import numpy as np
import os
import zipfile
import json
from os.path import join, dirname
from os import environ
from watson_developer_cloud import VisualRecognitionV3
import time
import datetime
import RPi.GPIO as GPIO

dir_path = os.path.dirname(os.path.realpath(__file__))
now = datetime.datetime.now()
realnow = now.strftime("%Y-%m-%d-%H-%M-%S")
foldername = ("%s/cropped/%s") %(dir_path, realnow)
caps_positions = []
results_json = []
images_folder = "%s" % (realnow)
os.makedirs(images_folder)
os.makedirs(foldername)
##############################
####### INIT CAMERAS #########
##############################

camera_pins = [4,14] # [2,3,4,14,15,17,18,27]
GPIO.setmode(GPIO.BCM)
for pin in camera_pins:
	GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
##############################
###### TAKE A PICTURE ########
##############################

def take_picture(image_number):

	print "Taking picture..."
	try: 
		cap = cv2.VideoCapture(0)
		cap.set(3,1280)
		cap.set(4,720)
		if not cap.isOpened():
			while not cap.isOpened():
				print "Cap not open... trying to fix it..."
				cap.open()
				time.sleep(1)
		else:
			time.sleep(1)
			ret, frame = cap.read()
			# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cv2.imwrite('%s/image_%s.png' % (images_folder,image_number),frame)
			cap.release()
			print "Picture taken"
	except Exception as e:
		print "Oops! something went wrong %s" % (e)

##############################
######## UNDISTORT ###########
##############################

def undistort_image(src):

	width = src.shape[1]
	height = src.shape[0]
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
	dst = cv2.undistort(src,cam,distCoeff)
	return dst


##############################
####### PROCESS IMAGE ########
##############################

def process_image(src):
	print "Processing image..."
	img_for_cropping = cv2.imread("%s/%s" %(images_folder, src))
	img_for_cropping = cv2.resize(img_for_cropping, (800,450), cv2.INTER_AREA)
	img_for_cropping = undistort_image(img_for_cropping)
	img = cv2.imread("%s/%s" %(images_folder, src),0)
	img = cv2.resize(img, (800,450), cv2.INTER_AREA)
	img = undistort_image(img)
	# cv2.imshow('dst', img)
	height, width = img.shape
	img = cv2.medianBlur(img,21)
	img = cv2.blur(img,(1,1))
	img = cv2.Canny(img, 0, 23, True)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)

	print "Detecting circles..."
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,150, param1=70,param2=28,minRadius=30,maxRadius=80)
	circles = np.uint16(np.around(circles))

	for i in circles[0,:]:
	    # crop cap
	    global caps_positions
	    caps_positions.append((i[0],i[1]))
	    margin = 30
	    originX = int(i[0])-int(i[2])-margin
	    originY = int(i[1])-int(i[2])-margin
	    endPointH = int(i[1])+int(i[2])+margin
	    endPointW = int(i[0])+int(i[2])+margin

	    if originX <= 0:
	        originX = 0
	    if originY <= 0: 
	        originY = 0
	    if endPointH >= height:
	        endPointH = height
	    if endPointW >= width:
	        endPointW = width

	    crop_img = img_for_cropping[originY:endPointH, originX:endPointW]

	    cv2.imwrite('%s/image_%s_%s.png'%(foldername, i[0], i[1]),crop_img)

	    # draw the outer circle
	    cv2.circle(img_for_cropping,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv2.circle(img_for_cropping,(i[0],i[1]),2,(0,0,255),3)
	    print len(circles)

	# cv2.imshow('detected circles',img_for_cropping)
	cv2.destroyAllWindows()
	print caps_positions
	print "Processing image done"

def compress_folder():
	print "Compressing folder..."
	myzip = zipfile.ZipFile("%s.zip"%(foldername), "w", zipfile.ZIP_DEFLATED)
	for root, dirs, files in os.walk(foldername):
	    for file in files:
	        myzip.write(os.path.join(root, file))
	myzip.close
	print "Folder compressed"



##############################
###### SEND TO WATSON ########
##############################

def run_nn():
	#  michelle: 0ba27a1a79d9d2f600ad71cc3c32fada1499a3a2
	#  joao: 24f5aba0d5d54d4ecc619de28e71ddfca61c7559
	visual_recognition = VisualRecognitionV3('2016-05-20', api_key='04b86ee445d9781762e7759131f130bcce64b45a')

	"Uploading to the Neural Network..."
	with open("%s.zip"%(foldername), 'rb') as image_file:
		results = json.dumps(visual_recognition.classify(images_file=image_file,  classifier_ids=['beercapz_620319077'], threshold=0.99), indent=2)
	print results
	global results_json
	results_json = results

	with open('output.json', 'w') as file_:
		file_.write(results)

##############################
######### DATA VIZ ###########
##############################

def data_viz(list_of_x, list_of_y, list_of_names):

	img = np.zeros((450,800,3), np.uint8)
	font = cv2.FONT_HERSHEY_SIMPLEX

	for x, y, name in zip(list_of_x, list_of_y, list_of_names):

		img = cv2.circle(img, (int(x),int(y)),40, (255,255,255), -1)
		cv2.putText(img, name, (int(x)-30,int(y)+60), font, 0.5,(255,255,255),2,cv2.LINE_AA)

	cv2.imwrite('img',img)
	# cv2.waitKey(0)
	cv2.destroyAllWindows()


##############################
####### PROCESS DATA #########
##############################

def process_data():

	list_of_x = []
	list_of_y = []
	list_of_names = []

	with open('output.json') as json_data:
	    d = json.load(json_data)
	    
	for images in d['images']:
		image_name = images['image'].rsplit('/',1)[-1]
		image_y = image_name.rsplit('_',1)[-1]
		image_y = image_y.rsplit('.',1)[-2]
		image_x = image_name.rsplit('_',2)[-2]
		list_of_x.append(str(image_x))
		list_of_y.append(str(image_y))

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

	# data_viz(list_of_x,list_of_y,list_of_names)


##############################
######## NARRATIVE ###########
##############################
number = 0
for pin in camera_pins:
	GPIO.output(pin, GPIO.HIGH)
	time.sleep(2)
	take_picture(number)
	GPIO.output(pin, GPIO.LOW)
	time.sleep(1)
	number = number + 1

for filename in os.listdir("%s/" % (images_folder)):
	if filename.endswith(".png"):
		filename = str(filename)
		print filename
		process_image(filename)

compress_folder()
run_nn()
process_data()






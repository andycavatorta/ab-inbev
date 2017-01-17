import numpy as np
import cv2
import os
import zipfile
import json
from os.path import join, dirname
from os import environ
from classifier import Classifier
from os import walk
image_classifier = Classifier()
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
now = datetime.datetime.now()
realnow = now.strftime("%Y-%m-%d-%H-%M-%S")
foldername = ("%s/results/%s") %(dir_path, realnow)
caps_positions = []
results_json = []

##############################
###### TAKE A PICTURE ########
##############################

def take_picture():

	print("Taking picture...")
	try: 
		cap = cv2.VideoCapture(0)
		ret, frame = cap.read()
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('videocapture/videocapture1.png',frame)
		cap.release()
		print("Picture taken")
	except Exception as e:
		print("Oops! something went wrong %s" % (e))

##############################
######## UNDISTORT ###########
##############################

def undistort_image(src):

	width = src.shape[1]
	height = src.shape[0]
	distCoeff = np.zeros((4,1),np.float64)

	k1 = -8.0e-5; # negative to remove barrel distortion
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

def process_image():
	print("Processing image...")
	img_for_cropping = cv2.imread('videocapture/videocapture1.png')
	img_for_cropping = cv2.resize(img_for_cropping, (800,450), cv2.INTER_AREA)
	# img_for_cropping = undistort_image(img_for_cropping)
	img = cv2.imread('videocapture/videocapture1.png',0)
	img = cv2.resize(img, (800,450), cv2.INTER_AREA)
	# img = undistort_image(img)
	cv2.imshow('dst', img)
	height, width = img.shape
	img = cv2.medianBlur(img,21)
	img = cv2.blur(img,(1,1))
	img = cv2.Canny(img, 0, 23, True)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)

	print("Detecting circles...")
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,150, param1=70,param2=28,minRadius=30,maxRadius=80)
	circles = np.uint16(np.around(circles))
	os.makedirs(foldername)

	for i in circles[0,:]:
	    # crop cap
	    global caps_positions
	    caps_positions.append((i[0],i[1]))
	    margin = 15
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

	    cv2.imwrite('%s/cap_%s_%s.jpg'%(foldername, i[0], i[1]),crop_img)

	    # draw the outer circle
	    cv2.circle(img_for_cropping,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv2.circle(img_for_cropping,(i[0],i[1]),2,(0,0,255),3)
	    print(len(circles))

	cv2.imshow('detected circles',img_for_cropping)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print(caps_positions)
	print("Processing image done")
	

##############################
#### SEND TO TENSORFLOW ######
##############################
def run_tensorflow():
	f = []
	for (dirpath, dirnames, filenames) in walk(foldername):
		f.extend(filenames)
		break
	results = []
	for image in f:
		guess = image_classifier.guess_image(foldername+'/'+image)
		results.append(guess)
	print(results)
	write_to_json(results)


def write_to_json(results):
	results_json = json.dumps(results, indent=2)

	with open('output-tf.json', 'w') as file_:
		file_.write(results_json)


##############################
######### DATA VIZ ###########
##############################

def data_viz(list_of_x, list_of_y, list_of_names):

	img = np.zeros((450,800,3), np.uint8)
	font = cv2.FONT_HERSHEY_SIMPLEX

	for x, y, name in zip(list_of_x, list_of_y, list_of_names):

		img = cv2.circle(img, (int(x),int(y)),40, (255,255,255), -1)
		cv2.putText(img, name, (int(x)-30,int(y)+60), font, 0.5,(255,255,255),2,cv2.LINE_AA)

	cv2.imshow('img',img)
	cv2.waitKey(0)
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

	print("Budlights: %s | Stella: %s | Hoegaarden: %s | Budweiser: %s | Platinum: %s | Ultra: %s " % (budlight, stella, hoegaarden, budweiser, platinum, ultra))

	data_viz(list_of_x,list_of_y,list_of_names)


##############################
######## NARRATIVE ###########
##############################

take_picture()
process_image()
run_tensorflow()
# run_nn()
# process_data()






import numpy as np
import cv2
import os
import zipfile
import json
from os.path import join, dirname
from os import environ
from watson_developer_cloud import VisualRecognitionV3
import datetime

visual_recognition = VisualRecognitionV3('2016-05-20', api_key='24f5aba0d5d54d4ecc619de28e71ddfca61c7559')

dir_path = os.path.dirname(os.path.realpath(__file__))

now = datetime.datetime.now()
realnow = now.strftime("%Y-%m-%d-%H-%M-%S")
foldername = ("%s/results/%s") %(dir_path, realnow)
caps_positions = []


##############################
###### TAKE A PICTURE ########
##############################

def take_picture():

	print "Taking picture..."
	try: 
		cap = cv2.VideoCapture(0)
		ret, frame = cap.read()
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('videocapture/videocapture1.png',frame)
		cap.release()
		print "Picture taken"
	except Exception as e:
		print "Oops! something went wrong %s" % (e)


##############################
####### PROCESS IMAGE ########
##############################

def process_image():
	print "Processing image..."
	img_for_cropping = cv2.imread('videocapture/videocapture1.png')
	img_for_cropping = cv2.resize(img_for_cropping, (800,450), cv2.INTER_AREA)
	img = cv2.imread('videocapture/videocapture1.png',0)
	img = cv2.resize(img, (800,450), cv2.INTER_AREA)
	height, width = img.shape
	img = cv2.medianBlur(img,21)
	img = cv2.blur(img,(1,1))
	img = cv2.Canny(img, 0, 23, True)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)

	print "Detecting circles..."
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

	    cv2.imwrite('%s/platinum_%s_%s.png'%(foldername, i[0], i[1]),crop_img)

	    # draw the outer circle
	    cv2.circle(img_for_cropping,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv2.circle(img_for_cropping,(i[0],i[1]),2,(0,0,255),3)
	    print len(circles)

	cv2.imshow('detected circles',img_for_cropping)
	cv2.destroyAllWindows()
	print caps_positions
	print "Processing image done"
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
	"Uploading to the Neural Network..."
	with open("%s.zip"%(foldername), 'rb') as image_file:
		results = json.dumps(visual_recognition.classify(images_file=image_file,  classifier_ids=['beercaps_101725851']), indent=2)
	print results

##############################
######## NARRATIVE ###########
##############################

take_picture()
process_image()
run_nn()




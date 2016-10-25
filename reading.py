import json
import csv
import numpy as np
import cv2

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

print list_of_x
print list_of_y
print list_of_names

img = np.zeros((450,800,3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

for x, y, name in zip(list_of_x, list_of_y, list_of_names):

	img = cv2.circle(img, (int(x),int(y)),40, (255,255,255), -1)
	cv2.putText(img, name, (int(x)-30,int(y)+60), font, 0.5,(255,255,255),2,cv2.LINE_AA)

cv2.imshow('img',img)
cv2.waitKey(0)

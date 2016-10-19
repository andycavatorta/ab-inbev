import cv2
import numpy as np

img_for_cropping = cv2.imread('test-pictures/teste19.png')
img = cv2.imread('test-pictures/teste19.png',0)
height, width = img.shape
img = cv2.medianBlur(img,21)
img = cv2.blur(img,(1,1))
img = cv2.Canny(img, 0, 21, True)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,17,2)

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)



circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,150,
                            param1=100,param2=33,minRadius=34,maxRadius=90)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    # crop cap
    margin = 15
    originX = i[0]-i[2]-margin
    originY = i[1]-i[2]-margin
    endPointH = i[1]+i[2]+margin
    endPointW = i[0]+i[2]+margin

    if originX <= 0: 
    	originX = 0
    if originY <= 0: 
    	originY = 0
    if endPointH >= height: 
    	endPointH = height
    if endPointW >= width: 
    	endPointW = width

    crop_img = img_for_cropping[originY:endPointH, originX:endPointW]


    cv2.imwrite('results/calibresult_%s.png'%(i),crop_img)

# crop_img = img2[circles[0][0][1]-circles[0][0][2]-30:circles[0][0][1]+circles[0][0][2]+30, circles[0][0][0]-circles[0][0][2]-30:circles[0][0][0]+circles[0][0][2]+30]
# cv2.imwrite('calibresult.png',crop_img)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

img_for_cropping = cv2.imread('test-pictures/obaoba.png')
img = cv2.imread('test-pictures/obaoba.png',0)
height, width = img.shape
img = cv2.medianBlur(img,21)
img = cv2.blur(img,(1,1))
img = cv2.Canny(img, 0, 23, True)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,17,2)

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)



circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,160,
                            param1=100,param2=30,minRadius=38,maxRadius=80)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # crop cap
    margin = 15
    originX = int(i[0])-int(i[2])-margin
    originY = int(i[1])-int(i[2])-margin
    endPointH = int(i[1])+int(i[2])+margin
    endPointW = int(i[0])+int(i[2])+margin

    if originX <= 0:
        print "here" 
        originX = 0
    if originY <= 0:
        print "here Y" 
        originY = 0
    if endPointH >= height:
        print "here endH" 
        endPointH = height
    if endPointW >= width:
        print "here endW" 
        endPointW = width

    crop_img = img_for_cropping[originY:endPointH, originX:endPointW]


    cv2.imwrite('results/obaoba_%s.png'%(i),crop_img)

        # draw the outer circle
    cv2.circle(img_for_cropping,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_for_cropping,(i[0],i[1]),2,(0,0,255),3)

# crop_img = img2[circles[0][0][1]-circles[0][0][2]-30:circles[0][0][1]+circles[0][0][2]+30, circles[0][0][0]-circles[0][0][2]-30:circles[0][0][0]+circles[0][0][2]+30]
# cv2.imwrite('calibresult.png',crop_img)
cv2.imshow('detected circles',img_for_cropping)
cv2.waitKey(0)
cv2.destroyAllWindows()
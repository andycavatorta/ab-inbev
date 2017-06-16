import os
import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt

DISTORTION     = np.array([[-6.0e-5, 0.0, 0.0, 0.0]], np.float64)
MIN_SIZE       = 65
MAX_CONFIDENCE = 1.0

""" Calculate camera matrix used to compensate for lens distortion
Adapted from Junwei's code:
https://github.com/andycavatorta/ab-inbev/blob/bd5cdd94e55ede948598a7d74e950424c8ec08e4/cropping_code/UnWrapImage.py
"""
def calc_camera_matrix((height,width)):
    cam = np.eye(3, dtype=np.float32) # assume unit matrix for camera

    cam[0, 2] = width  / 2.0  # center x
    cam[1, 2] = height / 2.0  # center y
    cam[0, 0] = 10.0          # focal length x
    cam[1, 1] = 10.0          # focal length y

    return cam

""" Correct for variable illumination
Based on tutorial by Regis Clouard:
https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html
"""
def correct_illumination(img, dark, bright):
    tmp = (img-dark) / (bright-dark)
    c1  = cv2.mean(tmp)[:1]

    result = cv2.mean(img)[:1] * (c1/tmp)
    return cv2.convertScaleAbs(result, alpha=(255)) # return 8-bit image

def equalize_histogram(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(img)

    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    #cv2.imshow('sdf',thresh)
    #cv2.waitKey()

    return thresh

def process_split(img):

    def process_channel(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(img)
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 65, 17)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None)
        return opening

    b, g, r = map(process_channel, cv2.split(img))
    cv2.bitwise_and(b, g, b)
    processed = cv2.bitwise_and(b, r)
    processed = cv2.bilateralFilter(processed, 9, 100, 100)

    #cv2.imshow('sdf',processed)
    #cv2.waitKey()

    return processed

def mask_circles(img):
    mask = np.zeros(img.shape[:2], dtype='uint8')

    # detect circles
    img = cv2.Canny(img, 100, 200)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 60, param1=90, param2=30, minRadius=65, maxRadius=110)

    if circles is not None: 
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]: cv2.circle(mask, (i[0],i[1]), i[2]/2, 255, -1)

    return mask

def mask_blobs(gray):
    mask = np.zeros(gray.shape[:2], dtype='uint8')

    # detect blobs
    mser = cv2.MSER_create(_delta=4, _min_area=65, _max_area=14400, _max_variation=1.0)
    blobs, _ = mser.detectRegions(gray)

    # find circular blobs

    for blob in blobs:
        hull = cv2.convexHull(blob.reshape(-1, 1, 2))

        epsilon = 0.01*cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, epsilon, True)

        # select polygons with more than 9 vertices
        if len(poly) > 9: 
            cv2.polylines(mask, [blob], 1, 255, 1)

    return mask

class Parser():

    def __init__(self, interactive=None, save_visuals=None):

        self.interactive  = interactive  or False
        self.save_visuals = save_visuals or False

    def mask_beers(self, img, shelf, camera):
        # create masks to accumulate blobs detected by each pass
        mask_distorted = np.zeros(img.shape[:2], dtype = 'uint8')
        mask           = np.zeros(img.shape[:2], dtype = 'uint8')
        
        # distorted:

        # CLAHE and Otsu threshold
        #equalized = equalize_histogram(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        #mask_distorted += mask_blobs(equalized)
        #mask_distorted += mask_circles(equalized)

        # adaptive threshold
        processed = process_split(img)
        mask_distorted += mask_blobs(processed)
        #mask_distorted += mask_circles(processed)

        # undistorted:

        img_out = cv2.undistort(img, self.cam, DISTORTION)

        # CLAHE and Otsu threshold
        #equalized = equalize_histogram(cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY))
        #mask += mask_blobs(equalized)
        #mask += mask_circles(equalized)

        # adaptive threshold
        processed = process_split(img_out)
        mask += mask_blobs  (processed)
        #mask += mask_circles(processed)
        
        # undistort results from distorted image; sum with undistorted results
        mask += cv2.undistort(mask_distorted, self.cam, DISTORTION)

        #dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        #_, mask_final = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)

        return mask # np.uint8(mask_final)

    def find_beers(self, mask, vis):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if max(w, h) < MIN_SIZE: continue

            (cx,cy), radius = cv2.minEnclosingCircle(contour)
            center = (int(cx),int(cy))
            radius = int(radius)

            circlePoints = cv2.ellipse2Poly(center, (radius,radius), 0, 0, 360, 1)
            confidence = cv2.matchShapes(contour, circlePoints, 1, 0.0)

            if confidence > MAX_CONFIDENCE: continue
                
            result.append((x, y, w, h))

            if self.interactive | self.save_visuals:
                cv2.polylines(vis, [contour], 0, (0,0,255), 1)
                cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.circle(vis, center, radius, (0,255,0), 2)                      
                cv2.putText(vis, '%.3f' % confidence, center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
        
        if self.interactive: plt.imshow(vis), plt.show()
        return (result, vis)

    def parse(self, shelf, camera, filename, filename_50=None, filename_0=None):
        img = cv2.imread(filename)
        img_weighted = img.copy()
        
        self.cam = calc_camera_matrix(img.shape[:2])
        mask_final = np.zeros(img.shape[:2], dtype='uint8') # self.mask_beers(img, shelf, camera)

        if filename_50 is not None: 
            img_weighted = cv2.addWeighted(img_weighted, 0.5, cv2.imread(filename_50), 0.5, 0)
            mask_final += self.mask_beers(cv2.imread(filename_50), shelf, camera)
        if filename_0  is not None:
            img_weighted = cv2.addWeighted(img_weighted, 0.5, cv2.imread(filename_0 ), 0.5, 0)
            mask_final += self.mask_beers(cv2.imread(filename_0 ), shelf, camera)

        if filename_50 is not None or filename_0 is not None:
            mask_final += self.mask_beers(img_weighted, shelf, camera)

        img_out = cv2.undistort(img, self.cam, DISTORTION)
        beer_bounds, vis = self.find_beers(mask_final, img_out.copy())

        return beer_bounds, vis, img_out

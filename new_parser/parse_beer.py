import math

import os
import sys

import cv2
import numpy as np

""" Compensate for lens distortion
Adapted from Junwei's code:
https://github.com/andycavatorta/ab-inbev/blob/bd5cdd94e55ede948598a7d74e950424c8ec08e4/cropping_code/UnWrapImage.py
"""
def undistort_image(img):
    height, width, _ = img.shape

    distCoeff = np.array([[-6.0e-5, 0.0, 0.0, 0.0]], np.float64)
    cam = np.eye(3, dtype=np.float32) # assume unit matrix for camera

    cam[0, 2] = width  / 2.0  # center x
    cam[1, 2] = height / 2.0  # center y
    cam[0, 0] = 10.0          # focal length x
    cam[1, 1] = 10.0          # focal length y

    return cv2.undistort(img, cam, distCoeff)

""" Correct for variable illumination
Based on tutorial by Regis Clouard:
https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html
"""
def correct_illumination(img, dark, bright):
    tmp = (img-dark) / (bright-dark)
    c1  = cv2.mean(tmp)[:3]

    result = cv2.mean(img)[:3] * (c1/tmp)
    return cv2.convertScaleAbs(result) # return an 8-bit image

def find_beers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis  = img.copy()

    # threshold
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    # detect blobs
    mser = cv2.MSER_create(_delta=4, _min_area=800, _max_area=14400, _max_variation=1.0)
    blobs, _ = mser.detectRegions(thresh)

    # find circles

    img_width, img_height = gray.shape
    mask = np.zeros((img_width, img_height, 1), dtype = "uint8")

    for blob in blobs:
        hull = cv2.convexHull(blob.reshape(-1, 1, 2))

        epsilon = 0.01*cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, epsilon, True)

        # select polygons with more than 9 vertices
        if len(poly) > 9: 
            cv2.polylines(mask, [blob], 1, 255, 2)
    
    # merge overlapping regions
    _, filled, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = []

    for contour in filled:
        cv2.polylines(vis, [contour], 0, (0,0,255), 1)

        (x,y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)

        cv2.circle(vis, center, radius, (0,255,0), 2)

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        result.append((x, y, w, h))

        # compare shapes with CONTOURS_MATCH_I1
        circlePoints = cv2.ellipse2Poly(center, (radius,radius), 0, 0, 360, 1)
        confidence = cv2.matchShapes(contour, circlePoints, 1, 0.0)
        
        cv2.putText(vis, '%.3f' % confidence, center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
    
    cv2.imshow('result', vis)
    cv2.waitKey()

    return result

def crop_beers(img, beer_bounds):
    img_width, img_height, _ = img.shape
    result = []

    for rect in beer_bounds:
        x, y, w, h = rect

        x = max(x - w/6,         0)
        w = min(w + w/3, img_width)

        y = max(y - h/6,         0)
        h = min(h + h/3, img_height)

        cropped = img[y:y+h, x:x+w].copy()
        result.append(cropped)

    return result

if __name__== '__main__':
    
    data_dir = './_data/illumination/'
    in_dir   = './_data/ShelfB_Test_Images/'
    out_dir  = './out/'

    if len(sys.argv) >= 2:
        in_dir   = sys.argv[1] + '/'
        
    if len(sys.argv) >= 3:
        data_dir = sys.argv[2] + '/'

    dark_dir   = data_dir + 'dark/'
    bright_dir = data_dir + 'bright/'

    print 'reading dark images from %s'   % (dark_dir)
    dark_images   = { int(os.path.splitext(f)[0]) : cv2.imread(dark_dir   + f) 
        for f in os.listdir(dark_dir)   if f.endswith('jpg') | f.endswith('png') }

    print 'reading bright images from %s' % (bright_dir)
    bright_images = { int(os.path.splitext(f)[0]) : cv2.imread(bright_dir + f) 
        for f in os.listdir(bright_dir) if f.endswith('jpg') | f.endswith('png') }

    print 'reading input from %s' % (in_dir)
    files = [f for f in os.listdir(in_dir) if f.endswith('jpg') | f.endswith('png')]

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for f in files:
        name = os.path.splitext(f)[0]

        shelf  = name[0]
        camera = int(name[1:])

        img = undistort_image(cv2.imread(in_dir + f))
        corrected = correct_illumination(img, dark_images[camera], bright_images[camera])
        
        beer_bounds = find_beers(corrected)
        beer_images = crop_beers(img, beer_bounds)

        count = 0
        for cropped in beer_images:
            
            path = os.path.join(out_dir, '%s_%d.png' % (name, count))
            
            print 'writing %s' % (path)
            cv2.imwrite(path, cropped)
            
            count += 1

import math
import collections

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

INTERACTIVE = True

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
def correct_illumination(img, dark):
    img    = np.float64(img) /255.0
    dark   = np.float64(dark)/255.0

    result = (img - dark) + cv2.mean(dark)[:3]
    return cv2.convertScaleAbs(result, alpha=(255)) # return 8-bit image

def find_beers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # equalize histogram
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)

    # threshold
    blur = cv2.GaussianBlur(equalized, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    # detect blobs
    mser = cv2.MSER_create(_delta=4, _min_area=800, _max_area=14400, _max_variation=1.0)
    blobs, _ = mser.detectRegions(thresh)

    # find circles

    img_width, img_height = gray.shape
    mask = np.zeros((img_width, img_height, 1), dtype = 'uint8')

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
    if INTERACTIVE: vis = img.copy()

    for contour in filled:
        x, y, w, h = cv2.boundingRect(contour)
        result.append((x, y, w, h))

        if INTERACTIVE:
            cv2.polylines(vis, [contour], 0, (0,0,255), 1)
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)

            (x,y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)

            cv2.circle(vis, center, radius, (0,255,0), 2)      

            # compare shapes with CONTOURS_MATCH_I1
            circlePoints = cv2.ellipse2Poly(center, (radius,radius), 0, 0, 360, 1)
            confidence = cv2.matchShapes(contour, circlePoints, 1, 0.0)
            
            cv2.putText(vis, '%.3f' % confidence, center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
    
    if INTERACTIVE: plt.imshow(vis), plt.show()
    return result

def crop_beers(img, beer_bounds):
    img_width, img_height, _ = img.shape
    result = []

    for rect in beer_bounds:
        x, y, w, h = rect
        size = max(w, h)

        x = max(x    - size/8,         0)
        w = min(size + size/4, img_width)

        y = max(y    - size/8,          0)
        h = min(size + size/4, img_height)

        cropped = img[y:y+h, x:x+w].copy()
        result.append(cropped)

    return result

def print_usage():
    print 'usage: %s [options]\n' \
          '  options:\n'          \
          '    -i <path> of fridge images\n'     \
          '    -c <path> of dark calibration images\n'      \
          '    -o <path> to save cropped images\n'         \
          '    -b run in batch mode' % (sys.argv[0])

if __name__== '__main__':
    d = os.path.dirname(__file__)

    data_dir = os.path.join(d, '_data', 'illumination', 'dark')
    in_dir   = os.path.join(d, '_data', 'ShelfB_Test_Images')
    out_dir  = os.path.join(d, 'out')

    if len(sys.argv) < 2: print_usage()
    else:
        
        it = iter(range (1, len(sys.argv)))
        for i in it:

            if sys.argv[i] == '-b': INTERACTIVE = False

            elif sys.argv[i] == '-i':
                try: in_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            elif sys.argv[i] == '-o':
                try: out_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            elif sys.argv[i] == '-c':
                try: data_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            else: print_usage(), sys.exit()

    print 'reading dark images from %s' % (data_dir)

    dark_images = collections.defaultdict(dict)
    for f in os.listdir(data_dir):

        name = os.path.splitext(f)[0]
        dark_images[name[0]][int(name[1:])] = cv2.imread(os.path.join(data_dir, f))

    print 'reading input images from %s' % (in_dir)
    files = [f for f in os.listdir(in_dir) if f.endswith('jpg') | f.endswith('png')]

    if not os.path.isdir(out_dir): os.mkdir(out_dir)

    for f in files:
        name = os.path.splitext(f)[0]

        shelf  = name[0]
        camera = int(name[:3][1:])

        img         = cv2.imread(os.path.join(in_dir, f))

        corrected   = correct_illumination(img, dark_images[shelf][camera])
        undistorted = undistort_image(corrected)
        
        beer_bounds = find_beers(undistorted)
        beer_images = crop_beers(undistorted, beer_bounds)

        count = 0
        for cropped in beer_images:
            
            path = os.path.join(out_dir, '%s_%d.png' % (name, count))
            
            print 'writing %s' % (path)
            cv2.imwrite(path, cropped)
            
            count += 1

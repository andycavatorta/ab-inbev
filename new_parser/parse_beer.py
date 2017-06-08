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
    (height, width) = img.shape[:2]

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
    c1  = cv2.mean(tmp)[:1]

    result = cv2.mean(img)[:1] * (c1/tmp)
    return cv2.convertScaleAbs(result, alpha=(255)) # return 8-bit image

def equalize_histogram(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(img)

    blur = cv2.GaussianBlur(equalized, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    return thresh


def mask_beers(gray, mask):
    # detect blobs

    mser = cv2.MSER_create(_delta=4, _min_area=800, _max_area=14400, _max_variation=1.0)
    blobs, _ = mser.detectRegions(gray)

    # find circles

    for blob in blobs:
        hull = cv2.convexHull(blob.reshape(-1, 1, 2))

        epsilon = 0.01*cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, epsilon, True)

        # select polygons with more than 9 vertices
        if len(poly) > 9: 
            cv2.polylines(mask, [blob], 1, 255, 2)

    return mask
    
def find_beers(mask, vis):
    # merge overlapping regions

    _, filled, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = []

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
    (img_height, img_width) = img.shape[:2]
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

    dark_dir   = os.path.join(d, '_data', 'illumination', 'dark')
    bright_dir = os.path.join(d, '_data', 'illumination', 'bright')
    in_dir     = os.path.join(d, '_data', 'ShelfB_Test_Images')
    out_dir    = os.path.join(d, 'out')

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
                try: dark_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            else: print_usage(), sys.exit()

    print 'reading dark images from %s' % (dark_dir)

    dark_images = collections.defaultdict(dict)
    for f in os.listdir(dark_dir):

        name = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(dark_dir, f))
        dark_images[name[0]][int(name[1:])] = img

    print 'reading bright images from %s' % (bright_dir)

    bright_images = collections.defaultdict(dict)
    for f in os.listdir(bright_dir):

        name = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(bright_dir, f))
        bright_images[name[0]][int(name[1:])] = img

    print 'reading input images from %s' % (in_dir)
    files = [f for f in os.listdir(in_dir) if f.endswith('jpg') | f.endswith('png')]

    if not os.path.isdir(out_dir): os.mkdir(out_dir)

    for f in files:
        name = os.path.splitext(f)[0].split('_')[0]

        shelf  = name[0]
        camera = int(name[1:])

        img_in  = cv2.imread(os.path.join(in_dir, f))
        img_out = undistort_image(img_in)

        (height, width) = img_in.shape[:2]
        mask = np.zeros((height, width, 1), dtype = 'uint8')
        
        # first pass: prospective illumination correction
        corrected = correct_illumination(img_in, dark_images[shelf][camera], bright_images[shelf][camera])
        mask = mask_beers(undistort_image(corrected), mask)

        # second pass: CLAHE and Otsu threshhold
        equalized = equalize_histogram(cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY))
        mask = mask_beers(equalized, mask)

        beer_bounds = find_beers(mask, img_out.copy())
        beer_images = crop_beers(img_out, beer_bounds)

        count = 0
        for cropped in beer_images:
            
            path = os.path.join(out_dir, '%s_%d.png' % (name, count))
            
            print 'writing %s' % (path)
            cv2.imwrite(path, cropped)
            
            count += 1

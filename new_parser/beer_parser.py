import os
import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt

DISTORTION = np.array([[-6.0e-5, 0.0, 0.0, 0.0]], np.float64)
MIN_SIZE = 65

""" Calculate camera matrix used to compensate for lens distortion
Adapted from Junwei's code:
https://github.com/andycavatorta/ab-inbev/blob/bd5cdd94e55ede948598a7d74e950424c8ec08e4/cropping_code/UnWrapImage.py
"""
def calc_camera_matrix(width, height):
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

    blur = cv2.GaussianBlur(equalized, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    return thresh

def process_split(img):      

    def process_channel(img):
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
        dilation = cv2.dilate(thresh, None, 1)
        return cv2.erode(dilation, None, 1)

    b, g, r = map(process_channel, cv2.split(img))
    cv2.bitwise_and(b, g, b)

    processed = cv2.bitwise_and(b, r)
    return cv2.GaussianBlur(processed, (7,7), 0)

def mask_blobs(gray, mask):
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

class Parser():

    def __init__(self, dark_dir, bright_dir, interactive=None, save_visuals=None):

        self.interactive  = interactive  or False
        self.save_visuals = save_visuals or False

        self.dark_images   = collections.defaultdict(dict)
        self.bright_images = collections.defaultdict(dict)

        print 'reading dark images from %s' % (dark_dir)
        for f in os.listdir(dark_dir):
            name = os.path.splitext(f)[0]
            img = cv2.imread(os.path.join(dark_dir, f))
            self.dark_images[name[0]][int(name[1:])] = img

        print 'reading bright images from %s' % (bright_dir)
        for f in os.listdir(bright_dir):
            name = os.path.splitext(f)[0]
            img = cv2.imread(os.path.join(bright_dir, f))
            self.bright_images[name[0]][int(name[1:])] = img

    def find_beers(self, mask, vis):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            size = max(w, h)
            if (size < MIN_SIZE):
                continue

            result.append((x, y, w, h))

            if self.interactive | self.save_visuals:
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
        
        if self.interactive: plt.imshow(vis), plt.show()
        return (result, vis)

    def parse(self, filename, shelf, camera):
        img_in  = cv2.imread(filename)
        
        (height, width) = img_in.shape[:2]
        cam = calc_camera_matrix(width, height)

        img_out = cv2.undistort(img_in, cam, DISTORTION)

        # create masks to accumulate blobs detected by each pass
        mask           = np.zeros((height, width), dtype = 'uint8')
        mask_distorted = np.zeros((height, width), dtype = 'uint8')
        
        # first pass: prospective illumination correction
        corrected = correct_illumination(img_in, self.dark_images[shelf][camera], self.bright_images[shelf][camera])
        mask = mask_blobs(cv2.undistort(corrected, cam, DISTORTION), mask)

        # second pass: CLAHE and Otsu threshold
        equalized = equalize_histogram(cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY))
        mask = mask_blobs(equalized, mask)

        # third pass: Adaptive threshold
        processed = process_split(img_out)
        mask = mask_blobs(processed, mask)

        # repeat each pass on distorted images

        corrected = correct_illumination(img_in, self.dark_images[shelf][camera], self.bright_images[shelf][camera])
        mask_distorted = mask_blobs(corrected, mask_distorted)

        equalized = equalize_histogram(cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY))
        mask_distorted = mask_blobs(equalized, mask_distorted)

        processed = process_split(img_in)
        mask_distorted = mask_blobs(processed, mask_distorted)

        # undistort results from distorted image; sum with undistorted results
        mask_final = mask + cv2.undistort(mask_distorted, cam, DISTORTION)

        beer_bounds, vis = self.find_beers(mask_final, img_out.copy())
        return beer_bounds, vis, img_out

import cv2
import os

import CropBC
import UnWrapImage

def unwarp_images_dir(in_dir, out_dir):
    if os.path.isdir(in_dir):
        files = os.listdir(in_dir)
        img_files = [f for f in files if f.endswith("png")]
        for img_file in img_files:
            img = cv2.imread(os.path.join(in_dir, img_file), cv2.IMREAD_COLOR)
            unwarped = UnWrapImage.undistort_image(image=img)
            cv2.imwrite(os.path.join(out_dir, img_file), unwarped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_images(in_dir, out_dir, tmp_dir):
    # min and max r vals just avg of junwei's original hardcoded vals
    minr = 74
    maxr = 96
    if os.path.isdir(out_dir):
        print("out already exists; maybe use a different one so you don't overwrite stuff?")
        return
    else:
        os.mkdir(out_dir)
    if os.path.isdir(tmp_dir):
        print("tmp dir already exists; maybe use a different one so you don't overwrite stuff?")
        return
    else:
        os.mkdir(tmp_dir)
    if os.path.isdir(in_dir):
        files = os.listdir(in_dir)
        img_files = [f for f in files if f.endswith("png")]
        for img_file in img_files:
            CropBC.crop(minr, maxr, in_dir, img_file, out_dir, tmp_dir)

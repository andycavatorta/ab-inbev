import cv2
import os
import sys

import CropBC
import UnWrapImage

def unwarp_images(in_dir, out_dir):
    if os.path.isdir(in_dir):
        files = os.listdir(in_dir)
        img_files = [f for f in files if f.endswith("png")]
        for img_file in img_files:
            img = cv2.imread(os.path.join(in_dir, img_file), cv2.IMREAD_COLOR)
            unwarped = UnWrapImage.undistort_image(image=img)
            cv2.imwrite(os.path.join(out_dir, img_file), unwarped)

def crop_images(in_dir, out_dir, tmp_dir):
    # min and max r vals just avg of junwei's original hardcoded vals
    minr = 65
    maxr = 100
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

def print_usage():
    print('usage: "%s unwarp <in_dir> <out_dir>" or "%s crop <in_dir> <out_dir> <tmp_dir>"' % (sys.argv[0], sys.argv[0]))

if __name__== '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == "unwarp":
            if len(sys.argv) == 4:
                unwarp_images(sys.argv[2], sys.argv[3])
            else:
                print_usage()
        elif sys.argv[1] == "crop":
            if len(sys.argv) == 5:
                crop_images(sys.argv[2], sys.argv[3], sys.argv[4])
            else:
                print_usage()
        else:
            print_usage()
    else:
        print_usage()

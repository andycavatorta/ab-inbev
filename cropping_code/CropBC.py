import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def circlerec(img, i, col, step):
    font = cv2.FONT_HERSHEY_SIMPLEX
    strstep = str(step)
    cv2.putText(img, strstep, (i[0], i[1]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)
    w = i[2]
    cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]),
                  (0, 255, 0), 5)


def continuedraw():
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # add ellipse pattern recognization


def crop(minr, maxr, img_in_dir, img_in_name, img_out_dir, tmp_dir):
    imgpath = os.path.join(img_in_dir, img_in_name)
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tmpimg = cv2.imread(imgpath)
    tmpimg = cv2.cvtColor(tmpimg, cv2.COLOR_BGR2RGB)
    row, col, rgb = img.shape

    # img = img[row / 10:(row * 9 / 10), (col / 4):(col * 3) / 4, :]

    print('Image shape is : ', img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # hough transform
    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                60, param1=90, param2=30, minRadius=minr, maxRadius=maxr)


    if circles1 is not None:
        circles = circles1[0, :, :]
        circles = np.int16(np.around(circles))
        # circles = np.nint16(np.around(circles))
        num, dim = circles.shape

        cnt = int(num * 0.8)
        # at junwei's suggestion, cap instances at 20
        #XXX are we actually taking the best 20, or just an arbitrary "first 20"?
        cnt = min(cnt, 20)

        step = 0
        for i in circles[:]:
            if step == cnt:
                break

            tmpcrop = img[max(i[1] - i[2], 0):min(i[1] + i[2], row), max(i[0] - i[2], 0):min(i[0] + i[2], col)]
            # tmpcrop = img[max(i[0] - i[2], 0):min(i[1] - i[2], col), min(i[0] + i[2], 0):max(i[1] + i[2], col)]
            m, n, c = tmpcrop.shape
            if m > 0 and n > 0:
                circlerec(img, i, col, step)
                print ('center and radii  ', step, '   : ', i[0], i[1], i[2])
                print (
                    'width and height : ', max(i[0] - i[2], 0), min(i[0] + i[2], col), max(i[1] - i[2], 0),
                    min(i[1] + i[2], row))

                img_in_name_base = '.'.join(img_in_name.split('.')[:-1]) # just strip off extension
                img_out_path = os.path.join(img_out_dir, img_in_name_base
                        + '_' + str(step) + '.png')
                cv2.imwrite(img_out_path, tmpcrop)
                step = step + 1

            else:
                continue

    plt.subplot(121)
    plt.imshow(tmpimg)
    plt.title('Original')

    plt.subplot(122)
    plt.imshow(img)
    plt.title('Cropped')

    tmpimgpath = os.path.join(tmp_dir, img_in_name)
    plt.savefig(tmpimgpath)
    #plt.show()

    continuedraw()


def crop_orig(minr, maxr, index):
    imgpath = './UnShelfB/bottles_B' + str(index) + '.png'
    print(imgpath)
    img_out_dir = './ShelfBCrop'# + str(index) + '_' + str(step) + '.png'
    crop(minr, maxr, "./UnShelfB", "bottles_B" + str(index)
            + '.png', img_out_dir, "./ShelfBWhole")

def cropbottle():
    raduis = [[75, 100], [75, 100], [75, 100], [75, 90], [75, 95], [75, 95], [70, 90], [70, 100], [70, 90], [75, 90],
              [75, 95], [75, 110]]
    index = 0

    for tmp in raduis:
        minr = tmp[0]
        maxr = tmp[1]
        crop_orig(minr, maxr, index)
        index = index + 1


def cropbottlecans():
    raduis = [[65, 100], [65, 100], [65, 100], [65, 90], [65, 95], [65, 95], [65, 90], [65, 100], [65,95], [65, 90],
              [65, 95], [65, 110]]
    index = 0

    for tmp in raduis:
        minr = tmp[0]
        maxr = tmp[1]
        crop(minr, maxr, index)
        index = index + 1

def cropbc():
    cropbottle()
    cropcan()


#cropbottle()

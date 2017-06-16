# ab-inbev

## Image Classifier

### Install Tensorflow
```
// Mac OS X, CPU only, Python 3.4 or 3.5:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl
$ pip3 install --upgrade $TF_BINARY_URL

// For the Raspberry PI
$ wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v0.11.0/tensorflow-0.11.0-cp27-none-linux_armv7l.whl
$ sudo pip install tensorflow-0.11.0-cp27-none-linux_armv7l.whl

```

### Training

* download and un-tar the latest `tf_files` directory such as http://njoliat.the-nsa.org/tf_files_170615.tar.gz into the `image_classifier_combined` directory (i.e. you should have `<project-root>/image_classifier_combined/tf_files`.
* From the `image_classifier_combined` folder, run:

```
$ sudo python3 retrain.py \
--bottleneck_dir=tf_files/bottlenecks \
--model_dir=tf_files/inception \
--output_graph=tf_files/retrained_graph.pb \
--output_labels=tf_files/retrained_labels.txt \
--image_dir tf_files/training_images \
--how_many_training_steps 4000
```

### Classification

* the basic classification code is in `classifier_combined.py`.  the function `guess_image` takes a tensorflow session and an image as an argument, and returns a list of `(guess, confidence)` in descending order; i.e. the first guess is the best.
* you can see `guess_image` in action in `classify_images.py`.  this code takes a directory full of images and a threshold confidence value, and for all images classified above the threshold, it sorts them into directories (this is useful for iteratively building the training set.)
* A WORD OF CAUTION: if you are adding images to the training set, do not clobber old images by adding new ones with the same names!  conveniently, tensorflow doesn't care what the names of the training images are, just the names of their directories; therefore if you're adding a new set of images, it's good to give the set a unique label (or date etc) and add that into the filenames of the images prior to sorting them.

### Classifier Test

* usage: `python classifier_test.py <input_dir>`.  it will print the percent correctness, and then info about all errors.
* input dir must have beers pre-sorted by directory, just like the training set would.  so you should see something like
```
(tensorflow) [njoliat@eschaton image_classifier_combined]$ ls sorted_test_set
bottle_becks        bottle_corona      bottle_shocktop_pretzel    bottle_ultra     can_bud_light  other
bottle_bud_america  bottle_hoegaarden  bottle_shocktop_raspberry  can_bud_america  can_busch
bottle_bud_light    bottle_platinum    bottle_stella              can_bud_ice      can_natty
```
* a smallish test set (400 images) can be found at http://njoliat.the-nsa.org/sorted_test_set_170616.tar.gz

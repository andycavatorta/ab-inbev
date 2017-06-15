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
* you can see `guess_image` in action in `classify_images.py`.  this code takes a directory full of images and a threshold confidence value, and for all images classified above the threshold, it sorts them into directories (this is useful for sorting more images to iteratively build the training set.)

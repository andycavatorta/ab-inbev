# ab-inbev

## Tensorflow

### Install
```
// Mac OS X, CPU only, Python 3.4 or 3.5:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl

$ pip3 install --upgrade $TF_BINARY_URL
```

### Training

* Save the images in jpg format.
* Drop them into the *tf_files/caps* folder. Create a subfolder for each brand. The name of the subfolder will be used by the script as the label for the images.  
* From the *image_training* folder, run:

```
$ sudo python3 retrain.py \
--bottleneck_dir=tf_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=tf_files/inception \
--output_graph=tf_files/retrained_graph.pb \
--output_labels=tf_files/retrained_labels.txt \
--image_dir tf_files/caps
```
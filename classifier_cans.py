import sys
import tensorflow as tf
import os

class Classifier():

    def __init__(self):
        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile("image_classifier_all/tf_files/retrained_labels_cans.txt")]

    def guess_image(self, test_images_dir="image_classifier_all/test_images_cans"):
        files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".jpg")])
        results = []

        # Unpersists graph from file
        with tf.gfile.FastGFile("image_classifier_all/tf_files/retrained_graph_cans.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            
            for image in files:

                # Read in the image_data
                image_data = tf.gfile.FastGFile(test_images_dir + "/" + image, 'rb').read()
                #print(test_images_dir + "/" + image)

                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                
                predictions = sess.run(softmax_tensor, \
                         {'DecodeJpeg/contents:0': image_data})
                
                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

                #scores = [] # list of (name, confidence)
                #for node_id in top_k:
                #    human_string = self.label_lines[node_id]
                #    score = predictions[0][node_id]
                #    scores += score
                scores = [predictions[0][node_id] for node_id in top_k]
                
                results.append((self.label_lines[top_k[0]], scores))
            return results

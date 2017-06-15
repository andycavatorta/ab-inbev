import sys
import tensorflow as tf
import os

class Classifier():

    def __init__(self):
        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile("tf_files/retrained_labels.txt")]

        #XXX moved this from guess_images.. is that ok?
        # Unpersists graph from file
        with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')


    def guess_images(self, input_dir):
        input_images = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
        with tf.Session() as sess:
            return [(i, guess_image(sess, i)) for i in input_images]

    def guess_image(self, tf_session, image_path):
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = tf_session.graph.get_tensor_by_name('final_result:0')
        
        predictions = tf_session.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        scores = [(self.label_lines[node_id], predictions[0][node_id]) for node_id in top_k]

        return scores

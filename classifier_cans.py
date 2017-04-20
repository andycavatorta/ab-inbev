import sys
import tensorflow as tf
from os import walk

class Classifier():

	def __init__(self):

		# Loads label file, strips off carriage return
		self.label_lines = [line.rstrip() for line 
						   in tf.gfile.GFile("image_classifier_cans/tf_files/retrained_labels.txt")]


	def guess_image(self, foldername):
	
		files = [("i%02d.jpg" % i) for i in range(26)]
		results = []
                
		#for (dirpath, dirnames, filenames) in walk(foldername):
		#	files.extend(filenames)
		#	break
		print("Found " + str(len(files)) + " files")			

		# change this as you see fit
		# image_path = sys.argv[1]
		# image_path = img

		# Unpersists graph from file
		with tf.gfile.FastGFile("image_classifier_cans/tf_files/retrained_graph.pb", 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')

		with tf.Session() as sess:
			
			for image in files:

				# Read in the image_data
				image_data = tf.gfile.FastGFile(foldername + "/" + image, 'rb').read()
				print(foldername + "/" + image)

				# Feed the image_data as input to the graph and get first prediction
				softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
				
				predictions = sess.run(softmax_tensor, \
						 {'DecodeJpeg/contents:0': image_data})
				
				# Sort to show labels of first prediction in order of confidence
				top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

				for node_id in top_k:
					human_string = self.label_lines[node_id]
					score = predictions[0][node_id]
					print('%s (score = %.5f)' % (human_string, score))
				
				# print(self.label_lines[top_k[0]])
				results.append(self.label_lines[top_k[0]])
				
			return results

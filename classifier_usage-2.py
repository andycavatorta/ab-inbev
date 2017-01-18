from classifier_2 import Classifier
from os import walk
image_classifier = Classifier()

foldername = "cropped/2017-01-18-01-42-34"

f = []
for (dirpath, dirnames, filenames) in walk(foldername):
	f.extend(filenames)
	break
print("Found " + str(len(f)) + " files")

results = []
for image in f:
	guess = image_classifier.guess_image(foldername+'/'+image)
	results.append(guess)
print("done guessing")
print(results)
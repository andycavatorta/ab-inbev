from classifier import Classifier
image_classifier = Classifier()

foldername = "cropped/2017-01-18-01-42-34"

guess = image_classifier.guess_image(foldername)
print(guess)
from classifier import Classifier

image_classifier = Classifier()
guess = image_classifier.guess_image("image_classifier/test_images/test_019.jpg")
print(guess)
from classifier import Classifier

image_classifier = Classifier()
guess = image_classifier.guess_image("test_images/test_019.jpg")
print(guess)
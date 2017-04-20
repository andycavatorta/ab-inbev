from classifier_cans import Classifier

ic = Classifier()
guess = ic.guess_image("image_classifier_cans/test_images")
print(guess)

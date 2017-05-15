from classifier_cans import Classifier
import os

ic = Classifier()
can_dirs  = ["bud_light", "budweiser", "busch", "natty", "ultra"]
for can_dir in can_dirs:
    print("")
    can_images_dir = "image_classifier_all/test_images_cans/" + can_dir
    can_images = sorted([f for f in os.listdir(can_images_dir) if f.endswith(".jpg")])
    scores = ic.guess_image(can_images_dir)
    for i in range(len(scores)):
        top_guess, confidence = scores[i][0]
        er_str = ""
        can_type = can_dir.replace("_", " ")
        if top_guess != can_type:
            er_str = "***ERROR*** classified as " + top_guess
        print("%s %s confidence level %f %s" % (can_type, can_images[i], confidence, er_str))

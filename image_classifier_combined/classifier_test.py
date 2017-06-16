from classifier_combined import Classifier
import os
import sys
import tensorflow as tf
import time

# NOTE: input_dir must have these subdirectories, or a subset of them:
#   bottle_becks
#   bottle_bud_america
#   bottle_bud_light
#   bottle_corona
#   bottle_hoegaarden
#   bottle_platinum
#   bottle_shocktop_pretzel
#   bottle_shocktop_raspberry
#   bottle_stella
#   bottle_ultra
#   can_bud_america
#   can_bud_ice
#   can_bud_light
#   can_busch
#   can_natty
#   other


# this is how tensorflow rolls with naming
def dir_to_name(dir_):
    return dir_.replace("_", " ")

class BeerInfo:
    def __init__(self, img_path, beer_type, guesses):
        self.img_path = img_path
        self.beer_type = beer_type
        self.guesses = guesses

def get_guesses(input_dir, beers_info):
    classifier = Classifier()
    with tf.Session() as sess:
        beer_dirs = os.listdir(input_dir)
        i = 0
        for beer_type_dir in beer_dirs:
            beer_type_path = os.path.join(output_dir, beer_type_dir)
            beer_type_imgs = os.path.listdir(beer_type_path)
            for img in beer_type_imgs:
                if (i % 10) == 0:
                    print('processed %d images' % i)
                img_path = os.path.join(beer_type_path, img)
                guesses = classifier.guess_image(sess, img_path)
                beer_name = dir_to_name(beer_type_dir)
                beers_info.append(BeerInfo(img_path, beer_name, guesses))
                i += 1
    
def test_classifier(input_dir):
    beers_info = []
    get_guesses(input_dir, beers_info)
    num_imgs = len(beers.info)

    num_errors = 0
    for beer_info in beers_info:
        if beer_info.beer_type != guesses[0][0]:
            num_errors += 1
    percent_correctness = ((num_imgs - num_errors) / float(num_imgs)) * 100
    print("correctness = %.2f%" % percent_correctness)
    print("errors:")
    for beer_info in beers_info:
        best_guess = beer_info.guesses[0]
        if beer_info.beer_type != best_guess[0]:
            print("    %s classified as %s with %.2f% confidence (%s)" %
                    (beer_info.beer_type, best_guess[0], best_guess[1],
                        beer_info.img_path))

    #TODO there might be other stats that it'd be nice to print, e.g. num
    # mistakes (or %correct) at each of a variety of confidence intervals.

def print_usage():
    print('usage: %s [options]\n'   \
          '  options:\n'            \
          '    -i <path> of input directory\n' % (sys.argv[0]))

if __name__ == "__main__":
    in_dir = "classifier_test_input"
    if len(sys.argv) > 1:
        it = iter(range (1, len(sys.argv)))
        for i in it:
            elif sys.argv[i] == '-i':
                try: in_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()
            else: print_usage(), sys.exit()
    test_classifier(in_dir)

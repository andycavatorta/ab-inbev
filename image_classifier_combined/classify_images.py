from classifier_combined import Classifier
import os
import sys
import tensorflow as tf

def classify_stuff(input_dir, output_dir, confidence_threshold):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    classifier = Classifier()
    input_images = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    with tf.Session() as sess:
        i = 0
        for image in input_images:
            if (i%10) == 0:
                print 'processing %dth image' % i
            guesses = classifier.guess_image(sess, os.path.join(input_dir, image))
            # classifier_results = [(best guess, confidence), (next guess, confidence), ...]
            best_guess, confidence = guesses[0]
            print 'best guess, confidence = %s, %f' % (best_guess, confidence)
            if confidence > confidence_threshold:
                # move the thing.
                beer_dir = best_guess.replace(" ", "_")
                from_path = os.path.join(input_dir, image)
                to_dir = os.path.join(output_dir, beer_dir)
                if not os.path.isdir(to_dir):
                    os.mkdir(to_dir)
                to_path = os.path.join(to_dir, image)
                print("move %s to %s" % (from_path, to_path))
                os.rename(from_path, to_path)
            i += 1

def print_usage():
    print 'usage: %s [options]\n'                                               \
          '  options:\n'                                                        \
          '    -t <integer> percentage confidence threshold to move an image\n' \
          '    -i <path> of input directory\n' \
          '    -o <path> of output directory\n' % (sys.argv[0])

if __name__ == "__main__":
    in_dir = "classifier_input"
    out_dir = "classifier_output"
    threshold = 0.9
    if len(sys.argv) > 1:
        it = iter(range (1, len(sys.argv)))
        for i in it:
            if sys.argv[i] == '-t':
                try: threshold = float(sys.argv[it.next()])
                except StopIteration: print_usage(), sys.exit()

            elif sys.argv[i] == '-i':
                try: in_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            elif sys.argv[i] == '-o':
                try: out_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            else: print_usage(), sys.exit()
    classify_stuff(in_dir, out_dir, threshold)

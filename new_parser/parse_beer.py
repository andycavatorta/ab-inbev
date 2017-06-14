import os
import sys

import cv2
import beer_parser

def crop_beers(img, beer_bounds):
    (img_height, img_width) = img.shape[:2]
    result = []

    for rect in beer_bounds:
        x, y, w, h = rect
        size = max(w, h)

        x = max(x    - size/4,          0)
        w = min(size + size/2, img_width)

        y = max(y    - size/4,          0)
        h = min(size + size/2, img_height)

        cropped = img[y:y+h, x:x+w].copy()
        result.append(cropped)

    return result

def print_usage():
    print 'usage: %s [options]\n'                                           \
          '  options:\n'                                                    \
          '    -i  <path> of fridge images. format: A11[_?].png|jpg\n'      \
          '    -i3 <path> of "0" "50" "100" image directories\n'            \
          '    -c  <path> of "dark" and "bright" calibration directories\n' \
          '    -o  <path> to save cropped images\n'                         \
          '    -v  <path> to (optionally) save visualizations\n'            \
          '    -b  run in batch mode' % (sys.argv[0])

if __name__== '__main__':
    d = os.path.dirname(__file__)

    data_dir = os.path.join(d, '_data', 'illumination')
    in_dir   = os.path.join(d, '_data', 'ShelfB_Test_Images')
    out_dir  = os.path.join(d, 'out')

    interactive  = True
    save_visuals = False
    use_average  = False

    if len(sys.argv) < 2: print_usage()
    else:
        
        it = iter(range (1, len(sys.argv)))
        for i in it:

            if sys.argv[i] == '-b': interactive = False

            elif sys.argv[i] == '-i':
                try: in_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            elif sys.argv[i] == '-o':
                try: out_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()

            elif sys.argv[i] == '-c':
                try: data_dir = sys.argv[it.next()]
                except StopIteration: print_usage(), sys.exit()
            
            elif sys.argv[i] == '-i3':
                try:
                    in_dir = sys.argv[it.next()]
                    use_average = True
                except StopIteration: print_usage(), sys.exit()

            elif sys.argv[i] == '-v':
                try: 
                    vis_dir = sys.argv[it.next()]
                    save_visuals = True
                except StopIteration: print_usage(), sys.exit()

            else: print_usage(), sys.exit()

    parser = beer_parser.Parser(
        os.path.join(data_dir, 'dark'), 
        os.path.join(data_dir, 'bright'), 
        interactive=interactive, save_visuals=save_visuals)

    if use_average:
        in_dir_50 = os.path.join(in_dir,  '50')
        in_dir_0  = os.path.join(in_dir,   '0')
        in_dir    = os.path.join(in_dir, '100')

    print 'reading input images from %s' % (in_dir)
    files = [f for f in os.listdir(in_dir) if f.endswith('jpg') | f.endswith('png')]

    if                  not os.path.isdir(out_dir): os.mkdir(out_dir)
    if save_visuals and not os.path.isdir(vis_dir): os.mkdir(vis_dir)

    for f in files:
        file_name = os.path.splitext(f)[0]
        names     = file_name.split('_')

        name    =       names[0]
        postfix = '_' + names[1] if len(names) > 1 else ''

        shelf   = name[0]
        camera  = int(name[1:])

        print 'processing %s' % (file_name)

        if not use_average:
            beer_bounds, vis, img_out = parser.parse(shelf, camera, 
                os.path.join(in_dir, f))
        else:
            beer_bounds, vis, img_out = parser.parse(shelf, camera,
                os.path.join(in_dir,    f),
                os.path.join(in_dir_50, f),
                os.path.join(in_dir_0,  f))

        beer_images = crop_beers(img_out, beer_bounds)

        if save_visuals:
             path = os.path.join(vis_dir, "%s_vis.png" % file_name)

             print 'writing visualization %s' % (path)
             cv2.imwrite(path, vis)

        count = 0
        for cropped in beer_images:
            
            path = os.path.join(out_dir, '%s%s_%d.png' % (name, postfix, count))
            
            print 'writing cropped %s' % (path)
            cv2.imwrite(path, cropped)
            
            count += 1

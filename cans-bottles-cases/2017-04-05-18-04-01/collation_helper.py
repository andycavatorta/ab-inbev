from PIL import Image
import os
import shutil

# move beers from "image_N/beer_type/" dir struct to "beer_type" dir struct, renaming to avoid collisions.
def collate_beers(beer_type):
    type_idx = 0
    for img_dir in ["image_1", "image_2", "image_3"]:
        src_loc = img_dir + "/" + beer_type
        for can_photo_name in os.listdir(src_loc):
            shutil.copyfile(src_loc + "/" + can_photo_name, "%s/i%02d.png" % (beer_type, type_idx))
            type_idx += 1

# just png to jpg conversion
def convert_img(beer_type):
    for png_file in os.listdir(beer_type):
        im = Image.open(beer_type + "/" + png_file)
        im.save(beer_type + "/" + png_file[:-3] + "jpg")

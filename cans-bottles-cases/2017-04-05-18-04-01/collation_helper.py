import os
import shutil

def collate_beers(beer_type):
    type_idx = 0
    for img_dir in ["image_1", "image_2", "image_3"]:
        src_loc = img_dir + "/" + beer_type
        for can_photo_name in os.listdir(src_loc):
            shutil.copyfile(src_loc + "/" + can_photo_name, "%s/i%02d.png" % (beer_type, type_idx))
            type_idx += 1

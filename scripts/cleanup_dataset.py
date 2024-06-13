from PIL import Image
import os
from tqdm import tqdm
import hashlib

imagedir = 'D:/DAYIII/MLErosionCustom/training_images/train'

def cleanup_dataset():
    for filename1 in tqdm(os.listdir(imagedir)):
            try:
                # img1 = Image.open(imagedir + "/" + filename1)
                im1 = hashlib.md5(open(imagedir + "/" + filename1, 'rb').read()).hexdigest()
                for filename2 in tqdm(os.listdir(imagedir)):
                    if filename1 == filename2:
                        continue
                    # img2 = Image.open(imagedir + "/" + filename2)
                    im2 = hashlib.md5(open(imagedir + "/" + filename2, 'rb').read()).hexdigest()
                    # if list(img1.getdata()) == list(img2.getdata()):
                    #     print("Identical")

                    if im1 == im2:
                        os.remove(imagedir + "/" + filename2)
                        print("Removed: ", filename2)
            except:
                continue

if __name__ == '__main__':
    cleanup_dataset()
import os
import cv2
from sunrainbowwhitehorse import face_enhance
from PIL import Image, ImageFilter

dirs = [r"C:\Users\hw\Desktop\Face_Recognition_Data\faces94\male",
        r"C:\Users\hw\Desktop\Face_Recognition_Data\faces94\female",
        r"C:\Users\hw\Desktop\Face_Recognition_Data\faces94\malestaff",
        r"C:\Users\hw\Desktop\Face_Recognition_Data\faces95",
        r"C:\Users\hw\Desktop\Face_Recognition_Data\faces96",
        r"C:\Users\hw\Desktop\Face_Recognition_Data\grimace"]
transdirs = [r"C:\Users\hw\Desktop\New_Face_Recognition_Data\faces94\male",
             r"C:\Users\hw\Desktop\New_Face_Recognition_Data\faces94\female",
             r"C:\Users\hw\Desktop\New_Face_Recognition_Data\faces94\malestaff",
             r"C:\Users\hw\Desktop\New_Face_Recognition_Data\faces95",
             r"C:\Users\hw\Desktop\New_Face_Recognition_Data\faces96",
             r"C:\Users\hw\Desktop\New_Face_Recognition_Data\grimace"]
k = 0
for dir in dirs:
    for person_name in os.listdir(dir):
        a = os.listdir(os.path.join(dir, person_name))
        print(len(a))
        if len(a) < 3:
            continue
        for i in a:
            img = Image.open(os.path.join(dir, person_name, i))
            # image = cv2.imread(os.path.join(dir, person_name, i))
            img = face_enhance(img)
            cv2.imwrite(str(os.path.join(transdirs[k], person_name, i)), img)
            os.system("python detect.py --source " + str(os.path.join(transdirs[k], person_name, i)) + " --target " +
                      str(os.path.join(transdirs[k], person_name)))
    k = k + 1

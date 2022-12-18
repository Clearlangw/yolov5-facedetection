import cv2
import dlib
import numpy as np
from PIL import Image, ImageFilter
import os
import platform
import sys
from pathlib import Path


def face_enhance(img):
    # enhance = img.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 大阈值边缘增强
    # result = cv2.cvtColor(np.asarray(enhance), cv2.COLOR_RGB2BGR)
    img = img.filter(ImageFilter.DETAIL)
    result = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) # 转化格式,cv2读取的是BGR
    gamma = 0.2
    scale = float(np.iinfo(result.dtype).max - np.iinfo(result.dtype).min)#计算范围
    result = ((result.astype(np.float32) / scale) ** gamma) * scale  # 归一化后自适应gamma增强
    result = np.clip(result, 0, 255).astype(np.uint8) # 转换回8位灰度

    return result


def show_recognition(img, org):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸检测画框
    detector = dlib.get_frontal_face_detector()
    # 获取人脸关键点检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # 获取人脸框位置信息
    dets = detector(gray, 1)  # 1表示采样（upsample）次数  0识别的人脸少点,1识别的多点,2识别的更多,小脸也可以识别

    for i in range(len(dets)):
        shape = predictor(img, dets[i])  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(org, pt_pos, 2, (0, 0, 255), 1)  # img, center, radius, color, thickness

    for i, d in enumerate(dets):
        print("第", i + 1, "个人脸的矩形框坐标：",
              "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())
        cv2.rectangle(org, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 0, 255), 2)


if __name__ == "__main__":
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    source = ROOT / 'data/preimages'
    target = ROOT / 'data/images'

    imgs = os.listdir(source)
    for imgpath in imgs:
        img = Image.open(str(source) + "/" + imgpath)
        org = cv2.imread(str(source) + "/" + imgpath)
        img = face_enhance(img)
        # show_recognition(img,org)
        cv2.imshow("image", org)
        cv2.imshow("newimage", img)
        imgpath = Path(imgpath)
        cv2.imwrite(str(target) + "/" + f'{imgpath.stem}' + ".jpg", img)

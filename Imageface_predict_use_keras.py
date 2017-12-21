# -*- coding: utf-8 -*-

import cv2
import sys
import gc
from face_train_use_keras import Model

MODEL_PATH = './me.face.model.h5'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # 加载模型
    model = Model()
    model.load_model(MODEL_PATH)


    faceID = model.face_predict(image)

    print (faceID)


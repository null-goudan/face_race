import cv2
import sys
import gc
from train_model import Model

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    #加载模型
    model = Model()
    model.load_model(file_path='./model/me.face.model.h5')

    #框住人脸的矩形边框颜色
    color = (0, 255, 0)

    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(int(sys.argv[1]), cv2.CAP_DSHOW)

    #人脸识别分类器本地存储路径
    cascade_path = cv2.data.haarcascades+"haarcascade_frontalface_alt2.xml"

    #循环检测识别人脸
    while True:
        ok, frame = cap.read()   #读取一帧视频
        if not ok:
            print('no')
            break

        #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                if faceRect[0] < 0 or faceRect[1] < 0 or faceRect[2] < 0 or faceRect[3] < 0:
                    pass
                x, y, w, h = faceRect
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame, 'Me',
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame, 'Other',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)

        cv2.imshow("Recognise myself", frame)

        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
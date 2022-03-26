import cv2
import sys
from PIL import Image


def draw_face(img, faceRects):
    if len(faceRects) > 0:  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 222, 111), 2)


def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_idx)

    classfier = cv2.CascadeClassifier(cv2.data.haarcascades+'/haarcascade_frontalface_alt2.xml')

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        draw_face(frame, faceRects)
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("Capturing Video", int(sys.argv[1]))

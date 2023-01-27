from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2

path = "a.png"
resp = RetinaFace.detect_faces(path)
print(len(resp.keys()))
img = cv2.imread(path)

for key in resp.keys():
    identity = resp[key]
    print(identity)
    facial_area = identity["facial_area"]
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255,0,0), 5)

plt.imshow(img[:, :, ::-1])
plt.show()


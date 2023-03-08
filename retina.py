from retinaface import RetinaFace
import cv2


MASKED_FRAMES_DIR = "masked_frames"


def mask_frame(path_to_frame):
    resp = RetinaFace.detect_faces("world-opencv" + "/" + path_to_frame)
    img = cv2.imread("world-opencv" + "/" + path_to_frame)
    try:
        count_faces = len(resp.keys())
        print(path_to_frame + f" has {count_faces} face")
        for key in resp.keys():
            identity = resp[key]
            print(identity)
            facial_area = identity["facial_area"]
            #cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255,0,0), 5)
            x1, y1, x2, y2 = facial_area
            face_img = img[y1:y2, x1:x2]  # Extract the face region
            # Apply Gaussian Blur to the face region
            blurred_face = cv2.GaussianBlur(face_img, (25, 25), 0)
            img[y1:y2, x1:x2] = blurred_face  # Replace the face region with the blurred face
        cv2.imwrite(MASKED_FRAMES_DIR + "/" + path_to_frame, img)

    except AttributeError:
        print(path_to_frame + " has no face")
        cv2.imwrite(MASKED_FRAMES_DIR + "/" + path_to_frame, img)




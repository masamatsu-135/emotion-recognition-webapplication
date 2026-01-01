from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

"""
class FaceBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def as_tuple(self):
        return (self.x, self.y, self.w, self.h)
"""

@dataclass
class FaceBox:
    """
    1つの顔領域を表す矩形 (x, y, w, h)
    x, y: 左上座標
    w, h: 幅と高さ
    """
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self):
        return (self.x, self.y, self.w, self.h)


def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
    return face_cascade


def detect_faces(image, bgr,
                 scale_factor=1.1, min_neighbors=5, min_size=(30,30)):
    
    boxes = []
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3), got {image.shape}")
    
    img = image.copy()

    if bgr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = get_face_cascade()
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor,
        minNeighbors=min_neighbors, minSize=min_size
        )
    """
    print(faces)
    print(type(faces))
    print(type(faces[0]))
    boxes.append(FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces)
    print(boxes)
    print(type(boxes))
    return faces
    """
    
    boxes: List[FaceBox] = [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    return boxes

def crop_faces(image, faces):

    crops = []
    h, w, _ = image.shape

    for face in faces:
        x1 = max(face.x, 0)
        y1 = max(face.y, 0)
        x2 = min(face.x + face.w, w)
        y2 = min(face.y + face.h, h)

        if x2 > x1 and y2 > y1:
            face_img = image[y1:y2, x1:x2, :]
            crops.append(face_img)

    return crops


def detect_and_crop_largest_face(image, bgr, **detect_kwargs):
    
    boxes = detect_faces(image, bgr=bgr, **detect_kwargs)
    print(boxes)
    print(type(boxes))
    if not boxes:
        return None, None
    print(f"boxes {boxes}")
    largest = max(boxes, key=lambda b: b.w * b.h)
    (face_img,) = crop_faces(image, [largest])

    return largest, face_img


if __name__ == "__main__":

    from pathlib import Path
    import os

    BASE_DIR = Path(__file__).resolve().parents[2]
    image_path = BASE_DIR / "tests" / "face.jpg"
    print(image_path)

    if image_path is None or not os.path.exists(image_path):
        print("error 1")

    else:
        img = cv2.imread(image_path)
        faces = detect_faces(img, bgr=True)
        print(f"Detected {len(faces)} face(s).")
        for i, face in enumerate(faces):
            print(f"Face {i}: {face}")

        largest_box, face_img = detect_and_crop_largest_face(img, bgr=True)
        if largest_box is not None:
            print("Largest face:", largest_box)

            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite("outputs/cropped_face.jpg", face_img)
            print("Saved cropped face")
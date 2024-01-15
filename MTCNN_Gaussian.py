import cv2
from mtcnn import MTCNN

def detect_and_blur_faces(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to read the image at path: {image_path}")
        return

    detector = MTCNN()
    faces = detector.detect_faces(image)
    print(str(len(faces)) + " total faces detected.")

    for face in faces:
        x, y, w, h = face['box']
        print(f"Face detected in the box {x} {y} {x+w} {y+h}")

        face_roi = image[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 15)
        image[y:y + h, x:x + w] = blurred_face

    # Resize the image before displaying
    scale_percent = 20  # You can adjust this value according to your needs
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Detected Faces with Blur', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'image3.jpg'
detect_and_blur_faces(image_path)


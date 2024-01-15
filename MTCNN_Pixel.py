import cv2
from mtcnn import MTCNN

def detect_and_process_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to read the image at path: {image_path}")
        return

    # Resize the image before face detection
    scale_percent = 20  # You can adjust this value according to your needs
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Detect faces using MTCNN on the resized image
    faces = detector.detect_faces(resized_image)
    print(str(len(faces)) + " total faces detected.")

    for face in faces:
        x, y, w, h = face['box']
        print(f"Face detected in the box {x} {y} {x+w} {y+h}")

        # Extract the face region
        face_roi = resized_image[y:y + h, x:x + w]

        # Resize the face region to (10, 10) using INTER_NEAREST interpolation
        resized_face = cv2.resize(face_roi, (10, 10), interpolation=cv2.INTER_NEAREST)

        # Resize the face region back to its original size using INTER_NEAREST interpolation
        resized_image[y:y + h, x:x + w] = cv2.resize(resized_face, (w, h), interpolation=cv2.INTER_NEAREST)

    # Display the result
    cv2.imshow('Detected Faces with Resizing', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'image3.jpg'
detect_and_process_faces(image_path)

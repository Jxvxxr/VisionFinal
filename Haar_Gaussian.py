import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_and_blur_faces(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"No hay imagen: {image_path}")
        return

    # Redimensiona la imagen para que sea más manejable en la pantalla
    scale_percent = 20  # Puedes ajustar este valor según tus necesidades
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(str(len(faces)) + " total faces detected.")

    for (x, y, w, h) in faces:
        print(f"Face detected in the box {x} {y} {x+w} {y+h}")
        face_roi = resized_image[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 15)
        resized_image[y:y + h, x:x + w] = blurred_face

    cv2.imshow('Detected Faces with Blur', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'image3.jpg'
detect_and_blur_faces(image_path)

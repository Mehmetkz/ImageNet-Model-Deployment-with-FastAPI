import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

image_files = [
    'apple.png',
    'clock.png',
    'oranges.png',
    'car.png'
]

for image_file in image_files:
    print(f"\nDisplaying image: {image_file}")
    img = mpimg.imread(f"imagenet/imagenet/{image_file}")
    imgplot = plt.imshow(img)
    plt.show()


def detect_and_draw_box(path):
    # Image --> numpy array
    img = cv2.imread(path)

    # Nesne tespit performansı
    bbox, label, conf = cv.detect_common_objects(img)

    # Mevcut resim dosyası
    print(f"***************\nImage processed: {path}\n")

    # Model çıktısı
    print(bbox, label, conf)

    # Modelin görüntü üzerinde gösterilmesi
    out = draw_bbox(img, bbox, label, conf)

    # Display the image with bounding boxes
    display(Image(out))
    plt.imshow(out)
    plt.show()

path = r'imagenet/imagenet/car.png'
detect_and_draw_box(path)

app = FastAPI(title='Deploying a ML Model with FastAPI')

class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected."


@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Desteklenmeyen dosya türü.")

    # Image stream
    image_stream = io.BytesIO(file.file.read())

    # İmlecin tekrar başa döndürülmesi
    image_stream.seek(0)

    # Numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # Numpy array -> image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


    # Modelin çalıştırılması
    bbox, label, conf = cv.detect_common_objects(image, model=model)

    # Tespit edilen nesnelerin box içerisinde gösterilmesi
    output_image = draw_bbox(image, bbox, label, conf)

    # Çıktıları kaydet
    cv2.imwrite(f'imagenet/Result_imagenet/{filename}', output_image)

    file_image = open(f'imagenet/imagenet/{filename}', mode="rb")

    return StreamingResponse(file_image, media_type="image/jpeg")


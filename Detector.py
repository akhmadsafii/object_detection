import cv2, time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
        
        # Colors list
        self.colorlist = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName,
        origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))

        print("Model " + self.modelName + " loaded successfully...")

    def createBoundigBox(self, image):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BAYER_BG2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = input[tf.newaxis,...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        if len(bboxs) != 0:
            for i in range(0, len(bboxs)):
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex]
                classColor = self.colorList[classIndex]

                displayText = '{}: {%}'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                print(ymin, xmin, ymax, xmax = bbox)
                break


    def predicImage(self, imagePath):
        image = cv2.imread(imagePath)

        self.createBoundigBox(image)

        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
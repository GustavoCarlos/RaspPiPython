###############################################################################
# Trabalho Robô detector de Faces
# Desenvolvido para a disciplina de inteligência artificial
# Universidade Federal de São Paulo - 2015
#
# Autor: Gustavo Carlos da Silva
# 
#
# Professor: Fábio Faria		
###############################################################################
#import the necessary packages
#Imports to use camera
from picamera.array import PiRGBArray
from picamera import PiCamera

import time
import cv2
import cv2.cv as cv
import numpy as np
import io

import os
import sys

#Import for hardware control
import RPi.GPIO as GPIO


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

#metodo para leitura das imagens de treino
def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
	dirnames.sort(key=int)
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

print 'Initializing Camera...'
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))

# allow the camera to warmup
time.sleep(0.1)

print 'Camera ready!'

#Controle de motores
print 'Motors control setting'
GPIO.setmode(GPIO.BOARD) #configura para numeracao de pinos da placa
GPIO.setup(19, GPIO.OUT) #pino 1 controle motor A
GPIO.setup(21, GPIO.OUT) #pino 2 controle motor A
GPIO.setup(22, GPIO.OUT) #pino 1 controle motor B
GPIO.setup(23, GPIO.OUT) #pino 2 controle motor B

#coloca todos os pinos em HIGH para nao girar motores pois sao ativos e LOW
GPIO.output(19, True)
GPIO.output(21, True)
GPIO.output(22, True)
GPIO.output(23, True)

#configura sinais PWM a 100 Hz
pwmMT1R = GPIO.PWM(19, 100) 
pwmMT1F = GPIO.PWM(21, 100)
pwmMT2F = GPIO.PWM(22, 100)
pwmMT2R = GPIO.PWM(23, 100)

#inicia todos a 100% de duty, motores parados
pwmMT1R.start(100)
pwmMT1F.start(100)
pwmMT2F.start(100)
pwmMT2R.start(100)

print 'Motors control ready!'


 

#Face recognizer part
print 'Reading images'
[X,y] = read_images(sys.argv[1])
print 'Images read complete'

#convert to 32bit num
y = np.asarray(y, dtype=np.int32)
print 'Creating and training model'

#model = cv2.createLBPHFaceRecognizer()
model = cv2.createEigenFaceRecognizer()
model.train(np.asarray(X), np.asarray(y))

print 'Model trained'



print 'Classifier Loading...'
#load the classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
print 'Classifier Loaded!'

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

	grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	grayImg = cv2.equalizeHist(grayImg)

	print 'Detecting faces'
	#code to detect faces
	#faces = face_cascade.detectMultiScale(grayImg, 1.1, 5)
	faces = face_cascade.detectMultiScale(grayImg, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
	print 'Squares of faces detected'
	print faces
	detectedFaceIndex = 100 #variable to verify who was detected
	#read keyboard
	for x, y, w, h in faces:
		face = grayImg[y:y+h, x:x+w]
		face_resized = cv2.resize(face,(92, 112))
		prediction = model.predict(face_resized) #get prediction
		print prediction
		(indexDt, accurace) = prediction 
		detectedFaceIndex = indexDt
		cv2.rectangle(image, (x, y),(x+w, y+h), (255, 0, 0), 2)
		draw_str(image, (x, y), 'Face detectada: ' + str(indexDt))
	##When GUI is working
	##cv2.imshow("Frame", image)
	##key = cv2.waitKey(1) & 0xFF

	#If face number 3 from gto set is detected
	if detectedFaceIndex == 3: #stop
		pwmMT1R.ChangeDutyCycle(100)
		pwmMT2F.ChangeDutyCycle(100)
	else: #move
		pwmMT1R.ChangeDutyCycle(40)
		pwmMT2F.ChangeDutyCycle(40)

	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break



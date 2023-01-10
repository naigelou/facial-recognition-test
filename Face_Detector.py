import cv2
import cvzone
import numpy as np

#Hakee treenidatan
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Tekee windowsista resizeablen.
#cv2.namedWindow('Niko',cv2.WINDOW_NORMAL)
#Kuva lukija
img = cv2.imread('shrek.png', cv2.IMREAD_UNCHANGED)

scale_precent = 10
width = int(img.shape[1]* scale_precent / 100)
height = int(img.shape[1]* scale_precent / 100)
dsize = (width,height)

output =cv2.resize(img,dsize)

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0),4)
        print(x,y)
        imgResult = cvzone.overlayPNG(frame,output,[x,y])
        cv2.imshow('Niko',imgResult)
        key = cv2.waitKey(1)
        
    if  key ==81 or key ==113:
        break
    
    
    
        
    

"""

#Muuttaa kuvan harmaaksi
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Tunnistaa naaman
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#Piirtää neliön
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),4)








#Displayaa kuvan näytölle, Eka parametri ottaa windowsnimen ja toinen kuvan.
cv2.imshow('Niko',img)
#Wait key pausee niin kauan että näppäintä on painettu.
cv2.waitKey()


print("Code completed")
"""
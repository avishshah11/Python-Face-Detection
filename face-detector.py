import cv2

#Some pre-trained data on face frontels from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in 
#img = cv2.imread('elon.png')
#to capture video from webcam
webcam = cv2.VideoCapture(0)

###read the current frame
while True:
    successful_frame_read, frame = webcam.read()

    #convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0, 255, 0), 2)

    cv2.imshow('User',frame)
    key = cv2.waitKey(1)

    #stop if Q key is pressed
    if key == 81 or key == 113:
        break
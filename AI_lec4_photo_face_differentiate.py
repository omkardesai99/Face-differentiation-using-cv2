import cv2
import os
import numpy as np

#this module cotains all fns that are in tester.py

def faceDetection(test_img):
    test_img=cv2.resize(test_img,(1000,1000))
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier(r'C:\Users\omkar desai\Desktop\Artificial Inteligence\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)
    
    return faces,gray_img

def labels_for_training_data(directory):
    faces=[] #ind
    faceId=[]  #dep
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print("skipping system file")#skip files/images that does not open
                
                continue
            
            id=os.path.basename(path)#fetching file names
            id

            img_path=os.path.join(path,filename)#fetiching file path
            print("img_path:",img_path)
            print('id:',id)
            #if id=='1:'
            test_img=cv2.imread(img_path)#load each img one by one
            
            if test_img is None:
                print("image not loaded properly!")
                continue
            faces_rect,gray_img=faceDetection(test_img)#calling facedetection func to return  faces detected in images
            
            if len(faces_rect)!=1:
                continue#since we are asuming only one person img is being fed to classifier as faces_rect are list in list[[1,2,3,4],[45,12,34,57]]
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]#corped the faces and stored here
            faces.append(roi_gray)
            faceId.append(int(id))
    return faces,faceId
  
         
#below fns trains haar classifier and takes faces,faceid returned by perveous function as its arguments
def train_classifier(faces,FaceId):
      face_recognizer=cv2.face.LBPHFaceRecognizer_create()
      face_recognizer.train(faces,np.array(FaceId))
      return face_recognizer



#below functions draw bounding boxes around detected face in img
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
    
    
#below function writes name of person for detected label
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)
    
'''
len(faces)==len(faceId)
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces,np.array(faceId))
face_recognizer=train_classifier(faces,faceId)
#face_recognizer.write('training.yml')
'''



























































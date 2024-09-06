import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands

hands = mpHand.Hands()

mpDraw = mp.solutions.drawing_utils

xmin,xmax = 600,640
ymin,ymax = 0,30

pTime = 0
cTime = 0

resize = False

while True:
    
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    cv2.rectangle(img,(xmax,ymin),(xmin,ymax),(0,0,255),-1)


    if results.multi_hand_landmarks:
        
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                
                h ,w ,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                
                if id in [4,8,12,16,20]:
                    cv2.circle(img, (cx, cy), 2, [0, 255, 0], cv2.FILLED)
                    
                    if (xmin <= cx <= xmax and ymin <= cy <= ymax):
                        cap.release()
    #fps
    cTime = time.time()   
    fps = 1/(cTime-pTime)
    pTime = cTime      
    cv2.putText(img, "FPS"+str(int(fps)), (10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
    cv2.imshow("img", img)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        
cv2.destroyAllWindows()
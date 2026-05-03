import cv2
print(cv2.__version__)
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.line(frame, (50,50), (200,200), (255,0,0), 3) 
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),2)
    cv2.putText(frame, "Press Q to quit", (100,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    overlay = frame.copy()
    h,w,_=frame.shape
    cv2.rectangle(overlay, (0,0), (w,h), (255,0,0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "Press Q to quit", (100,h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Test",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
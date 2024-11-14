import cv2

cap = cv2.VideoCapture(0)  # 0 untuk kamera web utama

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

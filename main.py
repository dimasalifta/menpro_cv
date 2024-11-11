import cv2

img = cv2.imread('images.jpg')

cv2.imshow('frame 1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
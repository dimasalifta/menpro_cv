import cv2

img = cv2.imread('rapip.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img,(200,200))



img_crop = img_resized[0:100,0:200]

cv2.line(img,(0,0),(400,400),(0,0,255),5) #BGR

cv2.line(img,(0,400),(400,0),(0,0,255),5) #BGR

cv2.circle(img,(200,130),100,(0,0,255),-1)
cv2.putText(img,'#ganyangrapip',(60,370),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)

cv2.imshow('frame 1', img)
# cv2.imshow('frame 2', img_gray)
# cv2.imshow('frame 3', img_resized)
# cv2.imshow('frame 4', img_crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
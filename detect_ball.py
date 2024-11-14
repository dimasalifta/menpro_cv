import cv2
import numpy as np

# Fungsi kosong untuk trackbar (diperlukan oleh OpenCV)
def nothing(x):
    pass

# Buka kamera (gunakan 0 untuk kamera utama)
cap = cv2.VideoCapture(0)

# Buat jendela untuk menampilkan video dan trackbar
cv2.namedWindow("Deteksi Bola")
cv2.namedWindow("Trackbars")

# Membuat trackbar untuk mengatur rentang HSV
cv2.createTrackbar("Lower Hue", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Lower Saturation", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("Lower Value", "Trackbars", 70, 255, nothing)
cv2.createTrackbar("Upper Hue", "Trackbars", 10, 180, nothing)
cv2.createTrackbar("Upper Saturation", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper Value", "Trackbars", 255, 255, nothing)

while True:
    # Baca frame dari video
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah frame ke ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Dapatkan nilai trackbar untuk batas bawah dan atas HSV
    lower_hue = cv2.getTrackbarPos("Lower Hue", "Trackbars")
    lower_saturation = cv2.getTrackbarPos("Lower Saturation", "Trackbars")
    lower_value = cv2.getTrackbarPos("Lower Value", "Trackbars")
    upper_hue = cv2.getTrackbarPos("Upper Hue", "Trackbars")
    upper_saturation = cv2.getTrackbarPos("Upper Saturation", "Trackbars")
    upper_value = cv2.getTrackbarPos("Upper Value", "Trackbars")

    # Set nilai batas bawah dan atas HSV
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])

    # Buat mask berdasarkan rentang HSV yang ditentukan
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Terapkan operasi morfologi untuk menghilangkan noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Temukan kontur pada mask untuk mendeteksi bola
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Abaikan kontur kecil
        if cv2.contourArea(contour) < 500:
            continue

        # Dapatkan lingkaran pembatas (enclosing circle) untuk kontur yang ditemukan
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # Gambar lingkaran di sekitar bola yang terdeteksi
        if radius > 10:  # Minimal ukuran radius untuk dianggap sebagai bola
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, "Bola Terdeteksi", (int(x) - 50, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan frame hasil dengan deteksi bola
    cv2.imshow("Deteksi Bola", frame)
    cv2.imshow("Mask", mask)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan objek VideoCapture dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()

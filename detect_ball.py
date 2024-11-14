import cv2
import numpy as np

# Buka kamera (gunakan 0 untuk kamera utama)
cap = cv2.VideoCapture(0)

# Tentukan rentang warna HSV untuk bola yang ingin dideteksi (contoh: bola berwarna biru)
# Sesuaikan nilai ini berdasarkan warna bola yang akan dideteksi
lower_hsv = np.array([100, 150, 70])  # Batas bawah HSV
upper_hsv = np.array([140, 255, 255]) # Batas atas HSV

while True:
    # Baca frame dari video
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah frame ke ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

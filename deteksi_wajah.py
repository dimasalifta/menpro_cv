import cv2

# Muat file Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Buka kamera (0 untuk kamera utama)
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari video
    ret, frame = cap.read()
    
    # Pastikan frame berhasil dibaca
    if not ret:
        break

    # Ubah frame menjadi grayscale (Haar Cascade lebih efektif dalam grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayy',gray)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(10, 10))

    # Gambarkan kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Tampilkan frame dengan deteksi wajah
    cv2.imshow('Deteksi Wajah', frame)

    # Tekan 'q' untuk keluar dari program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan objek VideoCapture dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()

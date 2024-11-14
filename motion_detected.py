import cv2

# Buka kamera (gunakan 0 untuk kamera utama)
cap = cv2.VideoCapture(0)

# Inisialisasi frame pertama untuk referensi
ret, frame1 = cap.read()
# Ubah ke grayscale dan gunakan sedikit blur untuk mengurangi noise
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

while True:
    # Baca frame berikutnya
    ret, frame2 = cap.read()
    if not ret:
        break
    
    # Ubah frame ke grayscale dan gunakan blur
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)
    
    # Hitung perbedaan absolut antara frame pertama dan kedua
    delta_frame = cv2.absdiff(frame1_gray, frame2_gray)
    
    # Terapkan thresholding untuk fokus pada area yang berubah
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate untuk menutup lubang kecil di area objek bergerak
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    
    # Temukan kontur pada frame yang sudah threshold
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambarkan persegi di sekitar area yang bergerak
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Abaikan area kecil (menghindari noise)
            
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame2,'Motion Detected',(60,370),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        
    # Tampilkan frame hasil dengan deteksi gerakan
    cv2.imshow("Deteksi Gerakan", frame2)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Delta Frame", delta_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set frame kedua sebagai frame pertama untuk iterasi berikutnya
    frame1_gray = frame2_gray

# Lepaskan objek VideoCapture dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()

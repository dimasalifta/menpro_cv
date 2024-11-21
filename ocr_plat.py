import cv2
import pytesseract

# Konfigurasi lokasi tesseract (ubah sesuai lokasi instalasi Anda)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Baca gambar menggunakan OpenCV
image_path = 'plat_putih2.jpg'  # Ganti dengan path gambar Anda
image = cv2.imread(image_path)

# Tampilkan gambar asli
cv2.imshow("Original Image", image)

# Preprocessing Gambar
# 1. Konversi ke Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray_image)
# 2. Binarisasi menggunakan Threshold
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Tampilkan hasil preprocessing
cv2.imshow("Processed Image", binary_image)

# OCR dengan Tesseract
custom_config = r"--oem 3 --psm 6"  # Mode default untuk teks terstruktur
extracted_text = pytesseract.image_to_string(binary_image, config=custom_config)


# Output teks yang dikenali
print("Hasil OCR:")
print(extracted_text)

# Tutup jendela OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()

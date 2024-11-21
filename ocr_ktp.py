import cv2
import pytesseract

# Konfigurasi lokasi tesseract (ubah sesuai lokasi instalasi Anda)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# OCR dengan Tesseract
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789"'  # Mode default untuk teks terstruktur
custom_config_text = r'--oem 3 --psm 6 -c tessedit_char_whitelist="qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"'  # Mode default untuk teks terstruktur

# Baca gambar menggunakan OpenCV
image_path = 'ktp_sample.jpg'  # Ganti dengan path gambar Anda
image = cv2.imread(image_path)
# Tampilkan gambar asliS
cv2.imshow("Original Image", image)

nik = image[110:200,200:660]
cv2.imshow("nik",nik)
pekerjaan = image[440:490,200:660]
cv2.imshow("pekerjaan",pekerjaan)

# Preprocessing Gambar
# 1. Konversi ke Grayscale
gray_image_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
# Tampilkan gambar asli
cv2.imshow("Gray Image1", gray_image_nik)
# 2. Binarisasi menggunakan Threshold
_, binary_image_nik = cv2.threshold(gray_image_nik, 128, 255, cv2.THRESH_BINARY)

# Tampilkan hasil preprocessing
cv2.imshow("Processed Image1", binary_image_nik)
#output
nik_extract = pytesseract.image_to_string(binary_image_nik, config=custom_config)
# Output teks yang dikenali
print("Hasil NIK:")
print(nik_extract)


# Preprocessing Gambar 2
# 1. Konversi ke Grayscale
gray_image_pekerjaan = cv2.cvtColor(pekerjaan, cv2.COLOR_BGR2GRAY)
# Tampilkan gambar asli
cv2.imshow("Gray Image2", gray_image_pekerjaan)
# 2. Binarisasi menggunakan Threshold
_, binary_image_pekerjaan = cv2.threshold(gray_image_pekerjaan, 128, 255, cv2.THRESH_BINARY)

# Tampilkan hasil preprocessing
cv2.imshow("Processed Image2", binary_image_pekerjaan)
#output
pekerjaan_extract = pytesseract.image_to_string(binary_image_pekerjaan, config=custom_config_text)
# Output teks yang dikenali
print("Hasil pekerjaan:")
print(pekerjaan_extract)


# Tutup jendela OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()

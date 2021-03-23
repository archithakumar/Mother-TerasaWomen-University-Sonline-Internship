import pytesseract
from PIL import Image
from pytesseract import *
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = Image.open("mh02fe8819.png")

output = pytesseract.image_to_string(img)

print(output)

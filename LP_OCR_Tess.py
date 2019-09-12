from PIL import Image
import pytesseract 
import argparse
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = r'D:\Study\Python_Scripts\Tesseract-OCR\tesseract.exe'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image to be OCR'd")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
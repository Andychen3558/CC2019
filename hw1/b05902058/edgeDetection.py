import sys
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt

def viewImage(img):
	cv2.imshow('Display', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

img = cv2.imread('leaf.jpg')
img_gray = cv2.imread('leaf.jpg', 0)

## detect and draw black edges
edged = cv2.Canny(img_gray, 30, 100)
rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR) ## 顏色空間轉換
output = img
output[np.where((rgb == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
## 另解
# output = np.bitwise_or(img, rgb)
# ret, threshold = cv2.threshold(rgb, 240, 255, cv2.THRESH_BINARY)
# output[threshold == 255] = 0 ## 把threshold為白點的位置在output上塗黑

# viewImage(output)

## detect external contour
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = 255 - gray
ret, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
contour = cv2.Canny(threshold, 30, 100)
# viewImage(contour)

## draw brown contours
contour = cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR)
output[np.where((contour == [255, 255, 255]).all(axis=2))] = [42, 42, 165]
# viewImage(output)

## save processed image
cv2.imwrite('output.png', output)

## draw school ID on processed image
img = Image.open('output.png')
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 12)
draw.text((160, 200), 'b05902058', font=font, fill='black')
img.save('output.png')

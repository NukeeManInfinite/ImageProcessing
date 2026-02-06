# import cv2
# import matplotlib.pyplot as plt
#
# img = cv2.imread("bg.jpg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# subtracted_img = cv2.subtract(img_rgb, 100)
#
# plt.subplot(1,2,1)
# plt.imshow(img_rgb)
# plt.title("Originial image")
# plt.axis("off")
#
# plt.subplot(1,2,2)
# plt.imshow(subtracted_img)
# plt.title("Subtracted image")
# plt.axis("off")
#
# plt.show()
# cv2.waitKey(0)

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("bg.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R, G, B =cv2.split(img_rgb)

R_addedImg = cv2.add(R, 150)

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("original image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(R_addedImg)
plt.title("Added intensity image")
plt.axis("off")

plt.show()



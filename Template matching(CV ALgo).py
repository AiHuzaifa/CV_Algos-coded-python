import cv2
import numpy as np


main_image = cv2.imread("main.png")
template = cv2.imread("template.png")

# 1. Convert to grayscale first (this removes the '3' channel number)
gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Convert pixels to floats so negative subtractions work perfectly
gray_main = gray_main.astype(np.float32)
gray_template = gray_template.astype(np.float32)

# 2. Extract and unpack the rows (height) and columns (width)
H_img, W_img = gray_main.shape

H_temp, W_temp = gray_template.shape

# 3. Print them to make sure the variables hold the right numbers
print(f"Main Image Height: {H_img}, Width: {W_img}")
print(f"Template Height: {H_temp}, Width: {W_temp}")


H_score = H_img - H_temp + 1
W_score = W_img - W_temp + 1


my_matrix = np.zeros((H_score, W_score), dtype=np.float32)

for h in range(H_score):
    for w in range(W_score):
        patch = gray_main[h : h + H_temp, w : w + W_temp]

        error = np.sum((patch-gray_template)**2)

        my_matrix[h, w] = error

# 1. Find the highest error number in your entire loop matrix
highest_error = np.max(my_matrix)

# 2. Divide by THAT specific maximum so your largest number becomes exactly 1.0
image_normalised = my_matrix / highest_error

# 3. Display it
cv2.imshow("myCHAIR", image_normalised)

cv2.waitKey(0)
cv2.destroyAllWindows()


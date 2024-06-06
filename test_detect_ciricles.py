import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

SIZE = 0.000018

# Read the image
filename = "img.png"
img = cv2.imread(filename)

# Convert the image from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Original Image")
plt.show()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform adaptive thresholding to segment the black regions
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Edge detection using Canny
edged = cv2.Canny(thresh, 30, 100)

# Find contours
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

screenCnt = None
list_approx = []

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, SIZE * peri, True)
    list_approx.append(approx)

# Use the largest approximate contour if no rectangle found
screenCnt = max(list_approx, key=cv2.contourArea) if list_approx else None

# Draw contour if detected
if screenCnt is not None:
    img_contour = img_rgb.copy()
    cv2.drawContours(img_contour, [screenCnt], -1, (0, 0, 255), 3)
    plt.imshow(img_contour)
    plt.title("Detected Contour")
    plt.show()

    # Create a mask and extract the contour area
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Convert mask to grayscale and detect edges for Hough Transform
    masked_gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

    # Find the longest line
    if lines is not None:
        longest_line = max(lines, key=lambda line: np.linalg.norm((line[0][2] - line[0][0], line[0][3] - line[0][1])))
        x1, y1, x2, y2 = longest_line[0]
        cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate the center of the longest line
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw the center point on the image
        cv2.circle(img_rgb, (center_x, center_y), 5, (255, 0, 0), -1)

        # Display the center coordinates
        print(f"Center of the longest line: ({center_x}, {center_y})")

    plt.imshow(img_rgb)
    plt.title("Detected Longest Line and Center")
    plt.show()

    # Convert mask to grayscale
    masked_gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=1, maxRadius=100)

    # Draw the detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img_rgb, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img_rgb, (x, y), 2, (255, 0, 0), 3)  # Draw center of circle

    plt.imshow(img_rgb)
    plt.title("Detected Circles in Contour")
    plt.show()
else:
    print("No contour detected")

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import imutils
from enum import Enum

SIZE = 0.000018
filename = "img_5.png"

# Predefined list of IDs based on circle color patterns
list_ids = {
    "P,P,G,P,B": 0,
    "G,P,G,P,B": 1,
    "G,G,G,P,B": 2,
    "P,G,G,P,B": 3,
    "P,P,P,G,B": 4,
    "G,P,P,G,B": 5,
    "G,G,P,G,B": 6,
    "P,G,P,G,B": 7,
    "G,G,G,G,B": 8,
    "P,P,P,P,B": 9,
    "P,P,G,G,B": 10,
    "G,G,P,P,B": 11,
    "G,P,G,G,B": 12,
    "G,P,P,P,B": 13,
    "P,G,G,G,B": 14,
    "P,G,P,P,B": 15,
}


class Position(Enum):
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    CENTER = 4


class Circle:
    def __init__(self, radius, color_name, center):
        self.radius = radius
        self.color = color_name
        self.x = center[0]
        self.y = center[1]


class Robot:
    def __init__(self, ID, center, angle):
        self.ID = ID
        self.center = center
        self.angle = angle


def get_color_name(h, s, v):
    pink_lower = np.array([140, 50, 50])
    pink_upper = np.array([170, 255, 255])
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])

    if pink_lower[0] <= h <= pink_upper[0] and pink_lower[1] <= s <= pink_upper[1] and pink_lower[2] <= v <= pink_upper[
        2]:
        return "P"
    elif green_lower[0] <= h <= green_upper[0] and green_lower[1] <= s <= green_upper[1] and green_lower[2] <= v <= \
            green_upper[2]:
        return "G"
    elif blue_lower[0] <= h <= blue_upper[0] and blue_lower[1] <= s <= blue_upper[1] and blue_lower[2] <= v <= \
            blue_upper[2]:
        return "B"
    else:
        return "Unknown"


def find_id(circles):
    detected_colors = ["Unknown"] * 5  # Initialize list for the five positions

    average_center_x = np.mean([circle.x for circle in circles])
    average_center_y = np.mean([circle.y for circle in circles])
    CENTER_POSITION = (0, 0)

    for circle in circles:
        if circle.y < average_center_y:
            if circle.x < average_center_x:
                detected_colors[Position.FIRST.value] = circle.color
            else:
                detected_colors[Position.SECOND.value] = circle.color
        else:
            if circle.x < average_center_x:
                detected_colors[Position.THIRD.value] = circle.color
            else:
                detected_colors[Position.FOURTH.value] = circle.color

        if abs(circle.x - average_center_x) < circle.radius and abs(circle.y - average_center_y) < circle.radius:
            detected_colors[Position.CENTER.value] = circle.color
            CENTER_POSITION = (circle.x, circle.y)

    pattern = ",".join(detected_colors)
    pattern_id = list_ids.get(pattern, "Unknown")
    return pattern_id, pattern, CENTER_POSITION



img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

plt.imshow(img_rgb)
plt.title("Original Image")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
edged = cv2.Canny(thresh, 30, 100)

contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

screenCnt = None
list_approx = []

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, SIZE * peri, True)
    list_approx.append(approx)

screenCnt = max(list_approx, key=cv2.contourArea) if list_approx else None

if screenCnt is not None:
    img_contour = img_rgb.copy()
    cv2.drawContours(img_contour, [screenCnt], -1, (0, 0, 255), 3)
    plt.imshow(img_contour)
    plt.title("Detected Contour")
    plt.show()

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    masked_gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    LINE_CENTER = (0, 0)
    if lines is not None:
        longest_line = max(lines, key=lambda line: np.linalg.norm((line[0][2] - line[0][0], line[0][3] - line[0][1])))
        x1, y1, x2, y2 = longest_line[0]
        cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        LINE_CENTER = (center_x, center_y)
        cv2.circle(img_rgb, (center_x, center_y), 5, (255, 0, 0), -1)
        print(f"Center of the longest line: ({center_x}, {center_y})")

    plt.imshow(img_rgb)
    plt.title("Detected Longest Line and Center")
    plt.show()

    circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=1,
                               maxRadius=100)

    detected_circles = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            color = get_color_name(hsv[y, x][0], hsv[y, x][1], hsv[y, x][2])
            detected_circles.append(Circle(r, color, (x, y)))
            print(f"Circle found at (x: {x}, y: {y}, radius: {r}), Color: {color}")

            cv2.circle(img_rgb, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img_rgb, (x, y), 2, (255, 0, 0), 3)

    pattern_id, pattern, CENTER_POSITION = find_id(detected_circles)
    print(f"Detected pattern: {pattern}, Pattern ID: {pattern_id}")
    dx = CENTER_POSITION[0] - LINE_CENTER[0]
    dy = CENTER_POSITION[1] - LINE_CENTER[1]

    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = (angle_degrees + 90) % 360 - 180
    robot = Robot(pattern_id, CENTER_POSITION, angle_degrees)

    print(f"Robot ID: {robot.ID}, Center: {robot.center}, Angle: {robot.angle}")
    plt.imshow(img_rgb)
    plt.title("Detected Circles in Contour")
    plt.show()
else:
    print("No contour detected")

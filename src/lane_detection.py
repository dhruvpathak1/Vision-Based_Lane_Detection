# Lane Detection Algorithm using Hough Transform
# Original Author: Dhruv Pathak
# Refactored for professional structure

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
INPUT_IMAGE_PATH = os.path.join("assets", "images", "lane_pic.jpg")
INPUT_VIDEO_PATH = os.path.join("assets", "videos", "test2.mp4")
OUTPUT_IMAGE_PATH = "lane_detection_test.jpg"
# Set to 'image' or 'video'
MODE = 'image'

# --- Functions ---

def canny(image):
    """Applies Gaussian Blur and Canny Edge Detection."""
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image


def region_of_interest(image):
    """Masks the image to focus only on the lane area."""
    height = image.shape[0]
    # Creating the region of interest with reference to plotted image
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


def coordinates(image, line_para):
    """Calculates coordinates for the line based on slope and intercept."""
    slope, intercept = line_para
    y1 = image.shape[0]
    # Length of the line
    y2 = int(y1 * (4 / 6))
    # Obtaining values of x1 and x2 from y= mx + c
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope(image, lines):
    """Averages the slopes and intercepts of detected lines to find main lanes."""
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # To determine the slope and intercept for a linear function of 1 degree
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        # Separating the 2 lines in 2 different arrays w.r.t slope
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Averaging out the values
    left_fit_avg = np.average(left_fit, axis=0) if left_fit else None
    right_fit_avg = np.average(right_fit, axis=0) if right_fit else None

    result_lines = []
    if left_fit_avg is not None:
        result_lines.append(coordinates(image, left_fit_avg))
    if right_fit_avg is not None:
        result_lines.append(coordinates(image, right_fit_avg))

    return np.array(result_lines)


def display_lines(image, lines):
    """Draws lines on a black image."""
    lines_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_img


def process_frame(frame):
    """Main pipeline for processing a single frame."""
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny_img = canny(gray_img)
    cropped_img = region_of_interest(canny_img)

    # Applying Hough Transform
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    avg_lines = average_slope(frame, lines)
    line_img = display_lines(frame, avg_lines)

    combo_image = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combo_image


def main():
    if MODE == 'image':
        if not os.path.exists(INPUT_IMAGE_PATH):
            print(f"Error: Image not found at {INPUT_IMAGE_PATH}")
            return

        org_img = cv2.imread(INPUT_IMAGE_PATH)
        # Resizing Image to 1290x705 as expected by the ROI logic
        org_img = cv2.resize(org_img, (1290, 705))

        final_image = process_frame(org_img)

        # Check if we have a display
        if os.environ.get('DISPLAY') is None:
            print(f"No display found. Saving output to {OUTPUT_IMAGE_PATH}")
            cv2.imwrite(OUTPUT_IMAGE_PATH, final_image)
        else:
            cv2.imshow("Lane Detection Result", final_image)
            print("Displaying result. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif MODE == 'video':
        if not os.path.exists(INPUT_VIDEO_PATH):
            print(f"Error: Video not found at {INPUT_VIDEO_PATH}")
            return

        vid = cv2.VideoCapture(INPUT_VIDEO_PATH)
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            # Resize frame to maintain consistency with ROI
            frame = cv2.resize(frame, (1290, 705))

            final_frame = process_frame(frame)

            if os.environ.get('DISPLAY') is None:
                # In headless mode, we might just process one frame and save it for testing
                print("No display found. Saving first frame to video_test.jpg and exiting.")
                cv2.imwrite("video_test.jpg", final_frame)
                break
            else:
                cv2.imshow("Lane Detection Video", final_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

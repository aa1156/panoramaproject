import cv2
import numpy as np

# Load the images to be stitched
img1 = cv2.imread('image2.jpg')
img2 = cv2.imread('image1.jpg')

# Initialize the keypoint detector and descriptor
orb = cv2.ORB_create()

# Find the keypoints and descriptors in both images
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Initialize the matcher and find the matches
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# Sort the matches by their distance
matches = sorted(matches, key=lambda x: x.distance)

# Keep only the top matches
good_matches = matches[:50]

# Extract the keypoints from the matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# Find the homography between the two images
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the second image to align with the first
result = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))

# Combine the two images into a panorama
result[:img1.shape[0], :img1.shape[1]] = img1
result[:img2.shape[0], img1.shape[1]:] = img2

# Save the result
cv2.imwrite('panorama.jpg', result)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse
import warnings


def key_points_matching_brute_force(features_first_img, features_second_img, method):

    bf = create_matching_object(method, crossCheck=True)
    best_matches = bf.match(features_second_img, features_first_img) 

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    return rawMatches

def key_points_matching_knn(features_first_img, features_second_img, ratio, method):
    bf = create_matching_object(method, crossCheck=False)
    rawMatches = bf.knnMatch(features_second_img, features_first_img, k=2) # Match descriptors.
    matches = []

    for m,n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches
    
def select_descriptor_methods(image, method=None):
    assert method is not None, "Please define a feature descriptor method. accepted Values are: 'sift', 'surf'"

    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    (keypoints, features) = descriptor.detectAndCompute(image, None)

    return (keypoints, features)
    
def create_matching_object(method, crossCheck):
    "Create and return a Matcher Object"

    # For BF matcher, first we have to create the BFMatcher object using cv2.BFMatcher().
    # It takes two optional params.
    # normType - It specifies the distance measurement
    # crossCheck - which is false by default. If it is true, Matcher returns only those matches
    # with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match
    # and vice-versa.
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def homography_stitching(keypoints_first_img, keypoints_second_img, matches, reprojThresh):
    # Converting the keypoints to numpy arrays before passing them for calculating Homography Matrix.
    # Because we are supposed to pass 2 arrays of coordinates to cv2.findHomography, as in I have these points in image-1, and I have points in image-2, so now what is the homography matrix to transform the points from image 1 to image 2
    
    keypoints_second_img = np.float32([keypoint.pt for keypoint in keypoints_second_img])
    keypoints_first_img = np.float32([keypoint.pt for keypoint in keypoints_first_img])

    if len(matches) > 4:
        # construct the two sets of points
        points_first = np.float32([keypoints_second_img[m.queryIdx] for m in matches])
        points_second = np.float32([keypoints_first_img[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(points_first, points_second, cv2.RANSAC, reprojThresh) # Calculate the homography between the sets of points

        return (matches, H, status)
    else:
        raise Exception("An error occurred during keypoints matching: probably the images are not part of the same scene.")



def main():

    parser = argparse.ArgumentParser(
                    prog='ImStitcher',
                    description='The utility that stitch two images',
                    epilog='Usage info')
                    
    parser.add_argument('-img1', '--left_img', required=True, help='Left image')
    parser.add_argument('-img2', '--right_img', required=True, help='Right image')
    parser.add_argument('-out', '--output', default='output.jpg', help='Output filename')
    parser.add_argument('-algo', '--algorithm', choices=['sift', 'brisk', 'orb'], default='sift', help='An algorithm name for image stitching')
    parser.add_argument('-flt', '--filter', choices=['knn', 'bf'], default='bf')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
     
    args = parser.parse_args()
    
    feature_extraction_algo = args.algorithm
    feature_to_match = args.filter

    first_img = cv2.imread(args.left_img)
    first_img = cv2.cvtColor(first_img,cv2.COLOR_BGR2RGB)
    first_img_gray = cv2.cvtColor(first_img, cv2.COLOR_RGB2GRAY)
    second_img = cv2.imread(args.right_img)
    second_img = cv2.cvtColor(second_img, cv2.COLOR_BGR2RGB)
    second_img_gray = cv2.cvtColor(second_img, cv2.COLOR_RGB2GRAY)

    keypoints_first_img, features_first_img = select_descriptor_methods(first_img_gray, method=feature_extraction_algo)
    keypoints_second_img, features_second_img = select_descriptor_methods(second_img_gray, method=feature_extraction_algo)
    
    fig = plt.figure(figsize=(15,10))
    plt.axis('off') 
        
    if(args.verbose):
        fig.add_subplot(2, 2, 1)
        plt.imshow(cv2.drawKeypoints(first_img_gray,keypoints_first_img,None,color=(0,255,0)))
        fig.add_subplot(2, 2, 2)
        plt.imshow(cv2.drawKeypoints(second_img_gray,keypoints_second_img,None,color=(0,255,0)))
        
    

    if feature_to_match == 'bf':
        matches = key_points_matching_brute_force(features_first_img, features_second_img, method=feature_extraction_algo)
        mapped_features_image = cv2.drawMatches(second_img, keypoints_second_img, first_img, keypoints_first_img, 
                                                matches[:100],
                                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if(args.verbose): 
            fig.add_subplot(2, 1, 2)
            plt.imshow(mapped_features_image)
            plt.show(block=False)
            
    elif feature_to_match == 'knn':
        matches = key_points_matching_knn(features_first_img, features_second_img, ratio=0.75, method=feature_extraction_algo)
        mapped_features_image_knn = cv2.drawMatches(second_img, keypoints_second_img, first_img, keypoints_first_img, 
                                                    np.random.choice(matches, 100),
                                                    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if(args.verbose): 
            fig.add_subplot(2, 1, 2)
            plt.imshow(mapped_features_image_knn)
            plt.show(block=False)

    (matches, homography_matrix, status) = homography_stitching(keypoints_first_img, keypoints_second_img, matches, reprojThresh=4)

    width = first_img.shape[1] + second_img.shape[1]
    height = max(first_img.shape[0], second_img.shape[0])


    result = cv2.warpPerspective(second_img, homography_matrix,  (width, height))
    result[0:first_img.shape[0], 0:first_img.shape[1]] = first_img

    imageio.imwrite(args.output, result)

    plt.figure(figsize=(15,10))
    plt.axis('off')
    plt.imshow(result)
    plt.show()


    
if __name__=="__main__":
    main()

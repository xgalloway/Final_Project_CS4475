#!/usr/bin/python

import os
import sys
import cv2
import math
import numpy as np
from PIL import Image
from numpy import linalg

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
                                 % cv2.__version__)
def imageDistance(matches):
    sumDistance = 0.0
    for match in matches:
        sumDistance += match.distance
    return sumDistance

def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:
        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
        hp_arr = np.array(hp, np.float32)
        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)
        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]
        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]
        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]
        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]
    min_x = min(0, min_x)
    min_y = min(0, min_y)
    return (min_x, min_y, max_x, max_y)

def stitchImages(base_img_rgb, dir_list, output, round, img_type):

    if ( len(dir_list) < 1 ):
        return base_img_rgb

    base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)

    # Use the SIFT feature detector
    detector = cv2.ORB_create()

    # Find key points in base image for motion estimation
    base_features, base_descs = detector.detectAndCompute(base_img, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    print "Iterating through next images..."

    closestImage = None

    # Find the best next image from the remaining images
    for next_img_path in dir_list:

        print "Reading %s..." % next_img_path

        if ( key_frame_file in next_img_path ):
            print "\t Skipping %s..." % key_frame
            continue

        # Read in the next image...
        next_img_rgb = cv2.imread(next_img_path)
        next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)

        print "\t Finding points..."

        # Find points in the next frame
        next_features, next_descs = detector.detectAndCompute(next_img, None)

        matches = matcher.match(next_descs, trainDescriptors=base_descs)

        print "\t Match Count: ", len(matches)

        matches = sorted(matches, key = lambda x:x.distance)
        distance = imageDistance(matches)

        print "\t Distance from Key Image: ", distance

        averagePointDistance = distance/float(len(matches))

        print "\t Average Distance: ", averagePointDistance

        kp1 = []
        kp2 = []

        for match in matches:
            kp1.append(base_features[match.trainIdx])
            kp2.append(next_features[match.queryIdx])

        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))

        inlierRatio = float(np.sum(status)) / float(len(status))

        # if ( closestImage == None or averagePointDistance < closestImage['dist'] ):
        if ( closestImage == None or inlierRatio > closestImage['inliers'] ):
            closestImage = {}
            closestImage['h'] = H
            closestImage['inliers'] = inlierRatio
            closestImage['dist'] = averagePointDistance
            closestImage['path'] = next_img_path
            closestImage['rgb'] = next_img_rgb
            closestImage['img'] = next_img
            closestImage['feat'] = next_features
            closestImage['desc'] = next_descs
            closestImage['match'] = matches

    print "Closest Image: ", closestImage['path']
    print "Closest Image Ratio: ", closestImage['inliers']

    new_dir_list = filter(lambda x: x != closestImage['path'], dir_list)

    H = closestImage['h']
    H = H / H[2,2]
    H_inv = linalg.inv(H)

    if ( closestImage['inliers'] > 0.1 ): # and 

        (min_x, min_y, max_x, max_y) = findDimensions(closestImage['img'], H_inv)

        # Adjust max_x and max_y by base img size
        max_x = max(max_x, base_img.shape[1])
        max_y = max(max_y, base_img.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if ( min_x < 0 ):
            move_h[0,2] += -min_x
            max_x += -min_x

        if ( min_y < 0 ):
            move_h[1,2] += -min_y
            max_y += -min_y

        print "Homography: \n", H
        print "Inverse Homography: \n", H_inv
        print "Min Points: ", (min_x, min_y)

        mod_inv_h = move_h * H_inv

        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))

        print "New Dimensions: ", (img_w, img_h)

        # Warp the new image given the homography from the old image
        base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))
        print "Warped base image"

        next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))
        print "Warped next image"

        # Put the base image on an enlarged palette
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        print "Enlarged Image Shape: ", enlarged_base_img.shape
        print "Base Image Shape: ", base_img_rgb.shape
        print "Base Image Warp Shape: ", base_img_warp.shape

        # Create a mask from the warped image for constructing masked composite
        (ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 
            0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, 
            mask=np.bitwise_not(data_map), 
            dtype=cv2.CV_8U)

        # Now add the warped image
        final_img = cv2.add(enlarged_base_img, next_img_warp, 
            dtype=cv2.CV_8U)

        # Crop off the black edges
        final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print "Found %d contours..." % (len(contours))

        max_area = 0
        best_rect = (0,0,0,0)

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            # print "Bounding Rectangle: ", (x,y,w,h)

            deltaHeight = h-y
            deltaWidth = w-x

            area = deltaHeight * deltaWidth

            if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
                max_area = area
                best_rect = (x,y,w,h)

        if ( max_area > 0 ):
            print "Maximum Contour: ", max_area
            print "Best Rectangle: ", best_rect

            final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                    best_rect[0]:best_rect[0]+best_rect[2]]

            final_img = final_img_crop

        # Write out the current round
        final_filename = "%s/%d." % (output, round)
        final_filename = final_filename + img_type
        cv2.imwrite(final_filename, final_img)

        return stitchImages(final_img, new_dir_list, output, round+1, img_type)

    else:

        return stitchImages(base_img_rgb, new_dir_list, output, round+1, img_type)

def resizeImages(dir_list, dir_name, width):
    width = int(width)
    for i in range(len(dir_list)):
        imTemp = Image.open(dir_list[i])
        wPercent = (width/float(imTemp.size[0]))
        height = int((float(imTemp.size[1]) * float(wPercent)))
        imTemp = imTemp.resize((width, height), Image.BILINEAR)
        imTemp.save(os.path.join(dir_name, dir_list[i]))

 # ----------------------------------------------------------------------------

# get user input
dir_name = raw_input('Enter images directory: ')
output_dir = raw_input('Enter output directory: ')
key_frame = raw_input('Enter key frame (full path): ')
width = raw_input('Enter image width (resize): ')
img_type = raw_input('Enter image type: ')

# Key frame
key_frame_file = key_frame.split('/')[-1]

# Open the directory given in the arguments
dir_list = []
try:
    dir_list = os.listdir(dir_name)
    dir_list = filter(lambda x: x.find(img_type) > -1, dir_list)

except:
    print >> sys.stderr, ("Unable to open directory: %s" % dir_name)
    sys.exit(-1)

dir_list = map(lambda x: dir_name + "/" + x, dir_list)
resizeImages(dir_list, dir_name, width)
dir_list = filter(lambda x: x != key_frame, dir_list)

base_img_rgb = cv2.imread(key_frame)

final_img = stitchImages(base_img_rgb, dir_list, output_dir, 0, img_type)


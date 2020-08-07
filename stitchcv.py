# import the necessary packages
from __future__ import print_function
import numpy as np
import datetime
import boto3
import botocore
import time
import cv2
import base64
import imutils

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X and initialize the
        # cached homography matrix
        self.isv3 = imutils.is_cv3(or_better=True)
        self.cachedH = None

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        # unpack the images
        (imageB, imageA) = images

        # if the cached homography matrix is None, then we need to
        # apply keypoint matching to construct it
        if self.cachedH is None:
            # detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                return None

            # cache the homography matrix
            self.cachedH = M[1]

        # apply a perspective transform to stitch the images together
        # using the cached homography matrix
        if(imageA.shape == None):
            imageA = imageA.next() 
        result = cv2.warpPerspective(imageA, self.cachedH, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None


DEFAULT_ENDPOINT = "http://ceph-route-rook-ceph.apps.jweng-ocp.shiftstack.com"
DEFAULT_ACCESS = "QjdOMFdZNEE3NTc3MUwwMDNZT1M="
DEFAULT_SECRET = "cmlBWFZLa2tIaWhSaTN5Sk5FNGpxaGRlc2ZGWWtwMWZqWFpqR0FrRA=="

#s3 = boto3.client(service_name = 's3', use_ssl = False, verify = False, endpoint_url=DEFAULT_ENDPOINT,
                           # aws_access_key_id = base64.decodebytes(bytes(DEFAULT_ACCESS,'utf-8')).decode('utf-8'),
                            #aws_secret_access_key = base64.decodebytes(bytes(DEFAULT_SECRET,'utf-8')).decode('utf-8'),)

# initialize the video streams and allow them to warmup
print("++++++ Getting video files...")
# leftStream = VideoStream(src=0).start()
# rightStream = VideoStream(usePiCamera=True).start()


#cap1 = s3.download_file(self.bucket,self.key,"out.mp4")
#cap2 = s3.download_file(self.bucket,self.key,"out.mp4")
captest1 = cv2.VideoCapture('first.avi')
captest2 = cv2.VideoCapture('second.avi')


time.sleep(2.0)

# initialize the image stitcher, motion detector, and total
# number of frames read
stitcher = Stitcher()
# motion = BasicMotionDetector(minArea=500)
total = 0

# loop over frames from the video streams
# 

while True:
    ret1, frame1 = captest1.read()
    ret2, frame2 = captest2.read()
    w = 400
    h = 400
    dim = (w, h)
    # frame1 = cv2.resize(frame1, dim)
    # frame2 = cv2.resize(frame2, dim)

    

    try:
        # frame1 = cv2.resize(frame1, dim)
        # frame2 = cv2.resize(frame2, dim)
        print("***************")
        if frame1 == None or frame2 == None:
            print("XXXXXX")
            break
        frame1 = imutils.resize(left, width=400)
        frame2 = imutils.resize(right, width=400)
        (h, w, d) = frame1.shape
        print("w: {}, h: {}, d: {}").format(w, h, d)
        
    except:
        print("error")
    
    
    
    
    res = stitcher.stitch([frame1, frame2])

    total += 1

    cv2.imshow("result", res)
    #cv2.imshow('left', frame1)
    #cv2.imshow("right", frame2)
    key = cv2.waitKey(1) & 0xFF
    print(total)
    if key == ord("q"): 
        break
'''    
while True:
    # grab the frames from their respective video streams
    # left = leftStream.read()
    # right = rightStream.read()
    ret1, frame1 = captest1.read()
    ret2, frame2 = captest2.read()

    # resize the frames
    # left = imutils.resize(left, width=400)
    # right = imutils.resize(right, width=400)
    w = 400
    h = 400
    dim = (w, h)
    frame1 = cv2.resize(frame1, dim) 
    frame2 = imutils.resize(frame2, dim)

    # stitch the frames together to form the panorama
    # IMPORTANT: you might have to change this line of code
    # depending on how your cameras are oriented; frames
    # should be supplied in left-to-right order
    
    result = stitcher.stitch([frame1, frame2])

    # no homograpy could be computed
    # if result is None: 
    #     print("[INFO] homography could not be computed")
    # 	break

    # # convert the panorama to grayscale, blur it slightly, update
    # # the motion detector
    # gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # locs = motion.update(gray)

    # # only process the panorama for motion if a nice average has
    # # been built up
    # if total > 32 and len(locs) > 0:
    # 	# initialize the minimum and maximum (x, y)-coordinates,
    # 	# respectively
    # 	(minX, minY) = (np.inf, np.inf)
    # 	(maxX, maxY) = (-np.inf, -np.inf)

    # 	# loop over the locations of motion and accumulate the
    # 	# minimum and maximum locations of the bounding boxes
    # 	for l in locs:
    # 		(x, y, w, h) = cv2.boundingRect(l)
    # 		(minX, maxX) = (min(minX, x), max(maxX, x + w))
    # 		(minY, maxY) = (min(minY, y), max(maxY, y + h))

    # 	# draw the bounding box
    # 	cv2.rectangle(result, (minX, minY), (maxX, maxY),
    # 		(0, 0, 255), 3)

    # increment the total number of frames read and draw the 
    # timestamp on the image
    
    total += 1
    # timestamp = datetime.datetime.now()
    # ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    # cv2.putText(result, ts, (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the output images
    cv2.imshow("Result", result)
    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)
    key = cv2.waitKey(1) & 0xFF

    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 
    # cap.get(cv2.CAP_PROP_FPS), (2 * frameWidth,frameHeight))
    # out.write(result)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
'''


# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
captest1.release()
captest2.release()

# leftStream.stop()
# rightStream.stop()

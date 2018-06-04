# python color_tracking.py --video balls.mp4

import argparse
from collections import deque

import cv2
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="Video4.mp4",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

lower = {'yellow': (0, 73, 173), 'red': (145, 82, 60)}
upper = {'yellow': (48, 186, 255), 'red': (255, 255, 255)}
colors = {'red': (0, 0, 255), 'yellow': (0, 255, 217)}
deque = {'yellow': deque(maxlen=args["buffer"]),
         'red': deque(maxlen=args["buffer"])}

pts = {'yellow': (0, 0), 'red': (0, 0)}
props = {'yellow': None, 'red': None}

# if a video path was not supplied, grab the reference
# to the webcam

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter("output.avi", fourcc, 30.0, (800, 450), True)

if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # IP webcam image stream
    # URL = 'http://10.254.254.102:8080/shot.jpg'
    # urllib.urlretrieve(URL, 'shot1.jpg')
    # frame = cv2.imread('shot1.jpg')

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=800)

    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # for each color in dictionary check object in frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for key, value in upper.items():
        # a series of dilations and erosions to remove any small
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(contours) > 0:
            # 경계값 중 먼저 찾은 경계값에서 가까운 경계값만 추적함
            for contour in contours:
                if props[key] is None:
                    props[key] = contours[0]
                else:
                    moment = cv2.moments(contour)
                    cx, cy = (int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00']))

                    # 미세한 조정 필요(속도, 물체개입 등)
                    if abs(cx - pts[key][0]) < 20 or abs(cy - pts[key][1]) < 20:
                        props[key] = contour
                    else:
                        continue

            # c = max(cnts, key=cv2.contourArea)
            (pts[key], radius) = cv2.minEnclosingCircle(props[key])
            M = cv2.moments(props[key])
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size. Correct this value for your obect's size
            if radius > 0.5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, center, int(radius), colors[key], 2)

        deque[key].appendleft(center)

        for i in range(1, len(deque[key])):
            if deque[key][i - 1] is None or deque[key][i] is None:
                continue

            thickness = int(np.sqrt(args["buffer"] / float(i + i)) * 1.5)
            cv2.line(frame, deque[key][i - 1], deque[key][i], colors[key], thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    writer.write(frame)

    key = cv2.waitKey(5) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
writer.release()
cv2.destroyAllWindows()

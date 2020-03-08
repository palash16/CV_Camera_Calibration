import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import math
import random

### Converting Euler Angles to Rotation Matrix
def eulerAnglesToRotationMatrix(theta) :
	R_x = np.array([[1,0,0],[0,math.cos(theta[0]),-math.sin(theta[0])],[0,math.sin(theta[0]), math.cos(theta[0])]])
	R_y = np.array([[math.cos(theta[1]),0,math.sin(theta[1])],[0,1,0],[-math.sin(theta[1]),0,math.cos(theta[1])]])             
	R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0],[math.sin(theta[2]),math.cos(theta[2]),0],[0,0,1]])
	R = np.dot(R_z,np.dot(R_y,R_x))
	return R

### Defining World Points
x,y=np.meshgrid(range(8),range(6))
x = x*29
y = y*29
world_points=np.hstack((x.reshape(48,1),y.reshape(48,1),np.zeros((48,1)))).astype(np.float32)
_3D_points = []
_2D_points = []

### Reading Images and image points
for i in range(5456,5471):
	img_path = 'Assignment1_Data/IMG_' + str(i) + '.JPG'
	I = cv2.imread(img_path)
	ret, corners = cv2.findChessboardCorners(I,(8,6))
	if ret is True:
		_3D_points.append(world_points)
		_2D_points.append(corners)

### Applying Zhang's Method
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3D_points, _2D_points, (I.shape[1],I.shape[0]), None, None)
print('Reprojection Error:', ret)
print ('Camera Calibration Matrix:')
print(mtx)
print('Distortion Parameters:')
print(dist)
print('Rotation Vectors for the images are:')
print(rvecs)
print('Translation Vectors for the images are:')
print(tvecs)
r =  rvecs[0]
R = (eulerAnglesToRotationMatrix(r))
world_points_1 = np.hstack((x.reshape(48,1),y.reshape(48,1),np.zeros((48,1)),np.ones((48,1)))).astype(np.float32)
temp1 = np.zeros((3,4))
temp1[0:3,0:3] = R[0:3,0:3]
temp1[:,3] = tvecs[0][:,0]
P = np.matmul(mtx,temp1)
P = P/P[2,3]
projected_points = []
for i in range(48):
	projection = np.matmul(P,np.transpose(world_points_1[i,:]))
	projection = projection/projection[2]
	projected_points.append(projection[0:2])
projected_points = np.asarray(projected_points)
I = cv2.imread('Assignment1_Data/IMG_5456.JPG')
plt.imshow(I)
idx = [7,15,23,31,39,47]
idx1 = [5,11,17,23,29,35,41,47]
q = 0
p = 0
for i in range(projected_points.shape[0]):
	if (i == idx[q]):
		q = q + 1
		continue
	plt.plot([projected_points[i][0],projected_points[i+1][0]],[projected_points[i][1],projected_points[i+1][1]],'go-')
for i in range(8):
	i1 = i
	j =  i + 8
	while(j < 48):
		plt.plot([projected_points[i1][0],projected_points[j][0]],[projected_points[i1][1],projected_points[j][1]],'go-')
		i1 = j
		j = j + 8
plt.imshow(I)
plt.title('Wireframe on reprojected points')
plt.show() 
I1 = cv2.imread('Assignment1_Data/IMG_5456.JPG')
I_undistort = cv2.undistort(I1,mtx,dist)
plt.subplot(1,2,1)
plt.imshow(I1)
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(I_undistort)
plt.title('Undistorted Image')
plt.show()
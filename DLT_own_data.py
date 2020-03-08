import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import math
import random


### Finding Projection Matrix using DLT
def projection_matrix_estimation(img_pts, world_pts):
    n = world_pts.shape[0]
    A =  np.zeros((2*n,12))
    for i in range(n):
        A[i*2,0:4] = -1 * world_pts[i,:]
        A[i*2,8:12] = img_pts[i,0]*world_pts[i,:]
        A[i*2+1,4:8] = -1 * world_pts[i,:]
        A[i*2+1,8:12] = img_pts[i,1]*world_pts[i,:]
    
    U, D, V = np.linalg.svd(A)
    P = V[11,:]
    P = (np.reshape(P,(3,4)))
    ### P is the projection matrix
    P = P/P[2,3]
    return P


### QR Decomposition
def DLT_algorithm(P):
    temp = np.linalg.inv(P[0:3,0:3])
    R,K = np.linalg.qr(temp)
    R = np.linalg.inv(R)
    K = np.linalg.inv(K)
    K = K/K[2,2]
    T = -1*np.matmul(temp,P[:,3])
    return R,K,T

### Reprojection Error for RANSAC
def reprojection_error(P,I_pts,W_pts):
    param = 10
    inliers = 0
    n = I_pts.shape[0]
    for i in range(n):
        projected_points = np.matmul(P,np.transpose(W_pts[i,:]))
        projected_points = projected_points/projected_points[2]
        error = np.abs(projected_points[0] - I_pts[i,0]) + np.abs(projected_points[1] - I_pts[i,1])
        if (error < param):
            inliers = inliers + 1
    return inliers

### RANSAC Algorithm
def RANSAC(img_points,world_points):
    N = 2000
    n = img_points.shape[0]
    I_pts = np.zeros((6,3))
    W_pts = np.zeros((6,4))
    I_rep = np.zeros((n-6,3))
    W_rep = np.zeros((n-6,4))
    current_best_inliers = 0
    best_projection_matrix = []
    for i in range(N):
        p = 0
        q = 0
        l  = random.sample(range(n),6)
        for j in range(n):
            if j in l:
                I_pts[q,:] = img_points[j,:]
                W_pts[q,:] = world_points[j,:]
                q = q + 1
            else:
                I_rep[p,:] = img_points[j,:]
                W_rep[p,:] = world_points[j,:]
                p = p + 1
        if (((np.sum(W_pts,axis=1))[1] == 0) or ((np.sum(W_pts,axis=1))[2] == 0)):
            continue
        P = projection_matrix_estimation(I_pts, W_pts)
        inl = reprojection_error(P,I_rep,W_rep)
        if (inl > current_best_inliers):
            best_projection_matrix =  P.copy()
            current_best_inliers = inl
            # print(current_best_inliers)
    return best_projection_matrix

### Main Code starts here
### Reading Image
I = cv2.imread('Own_data/DLT.jpg')
plt.imshow(I)
plt.title('Original Image')
plt.show()

### Manually defining Image Points
I_pts  = [[1261,474],[1587,474],[1911,474],[2239,474],[1274,788],[1593,788],[1906,788],[2220,788],
		[1289,1075],[1600,1075],[1901,1075],[2206,1075],[1280,1251],[1593,1251],[1904,1257],[2218,1257],
		[1240,1389],[1581,1389],[1920,1389],[2261,1394],[1189,1553],[1567,1551],[1938,1551],[2315,1555]]
I_pts = np.asarray(I_pts)

### Defining World Points
X = [15,10,5,0,15,10,5,0,15,10,5,0,15,10,5,0,15,10,5,0,15,10,5,0]
Y = [12.5,12.5,12.5,12.5,7.5,7.5,7.5,7.5,2.5,2.5,2.5,2.5,0,0,0,0,0,0,0,0,0,0,0,0]
Z = [0,0,0,0,0,0,0,0,0,0,0,0,2.5,2.5,2.5,2.5,7.5,7.5,7.5,7.5,12.5,12.5,12.5,12.5]
n = len(I_pts)
I1 = I.copy()
for pt in I_pts:
    I1[pt[1]-5:pt[1]+5,pt[0]-5:pt[0]+5] = [255,0,0] 
plt.imshow(I1)
plt.title('Original Image with marked Original Points')
plt.show()

world_pts = []
image_pts = np.zeros((n,3))

for i in range(n):
    image_pts[i,0:2] = I_pts[i,:]
    image_pts[i,2] = 1
    world_pts.append([X[i],Y[i],Z[i],1])
    
image_pts = np.asarray(image_pts)
world_pts = np.asarray(world_pts)

### Estimating Projection Matrix with all the points
P = projection_matrix_estimation(image_pts,world_pts)
print('Projection Matrix is given as :')
print(P)

### Finding Reprojected Points
projections = np.zeros((image_pts.shape[0],3))
for i in range(image_pts.shape[0]):
    projections[i,:] = np.matmul(P,np.transpose(world_pts[i,:]))
    projections[i,:] =  projections[i,:]/projections[i,2]
ppp = []
for aa in projections:
    ppp.append([int(aa[0]),int(aa[1])])


### Plotting the reconstructed and original points
reprojected_image = I.copy()
for pt in I_pts:
    reprojected_image[pt[1]-5:pt[1]+5,pt[0]-5:pt[0]+5,:] = [0,0,255]
for pt in ppp:
    reprojected_image[pt[1]-5:pt[1]+5,pt[0]-5:pt[0]+5,:] = [255,0,0]

plt.imshow(reprojected_image)
plt.title('DLT        Blue: Original Points        Red: Reprojected Points')
plt.show()


R,K,T = DLT_algorithm(P)
print('Camera Matirx is :')
print(K)
print('Rotation Matrix is :')
print(R)
print('Projection Center is:')
print(T)


### Estimating P Matrix with RANSAC Algorithm
P_best = RANSAC(image_pts,world_pts)
print('Projection Matrix after RANSAC is:')
print(P_best)
R_best,K_best,T_best = DLT_algorithm(P_best)
print('Camera Matirx is :')
print(K_best)
print('Rotation Matrix is :')
print(R_best)
print('Projection Center is:')
print(T_best)


### Finding Reprojected Points
projections = np.zeros((image_pts.shape[0],3))
for i in range(image_pts.shape[0]):
    projections[i,:] = np.matmul(P_best,np.transpose(world_pts[i,:]))
    projections[i,:] =  projections[i,:]/projections[i,2]
ppp = []
for aa in projections:
    ppp.append([int(aa[0]),int(aa[1])])


### Plotting the reconstructed and original points
reprojected_image = I.copy()
for pt in I_pts:
    reprojected_image[pt[1]-5:pt[1]+5,pt[0]-5:pt[0]+5,:] = [0,0,255]
for pt in ppp:
    reprojected_image[pt[1]-5:pt[1]+5,pt[0]-5:pt[0]+5,:] = [255,0,0]

plt.imshow(reprojected_image)
plt.title('RANSAC DLT      Blue: Original Points        Red: Reprojected Points')
plt.show()

### Plotting the Wireframe on the grid
wireframe = I.copy()
plt.imshow(wireframe)


idx = [3,7,11,15,19,23]
q = 0
for i in range(n-1):
    if i == idx[q]:
        q = q + 1
        continue
    plt.plot([ppp[i][0],ppp[i+1][0]],[ppp[i][1],ppp[i+1][1]],'go-')

for i in range(4):
	i1 = i
	j =  i + 4
	while(j < 24):
		plt.plot([ppp[i1][0],ppp[j][0]],[ppp[i1][1],ppp[j][1]],'go-')
		i1 = j
		j = j + 4
plt.title('Wireframe on reprojected points')
plt.show()
### Finding Radial Distortions
I_gray =cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
temp = [[-1,0,0],[0,-1,0],[0,0,1]]
K_positive = np.matmul(K_best,temp)
K_positive[0,1] = 0

world_pts1 = world_pts[:,:3]
image_pts1 = image_pts[:,:2]
world_pts1 = world_pts1.astype('float32')
image_pts1 = image_pts1.astype('float32')
ret, K_temp, dist, rvecs, tvecs = cv2.calibrateCamera([world_pts1],[image_pts1],I_gray.shape[::-1],K_positive,None,None,flags = (cv2.CALIB_USE_INTRINSIC_GUESS))

h, w = I.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K_temp,dist,(w,h),1,(w,h))

print('Distortion Parameters are:')
print(dist)
I_undistort = cv2.undistort(I,K_temp,dist,None,newcameramtx)
plt.subplot(1,2,1)
plt.imshow(I)
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(I_undistort)
plt.title('Undistorted Image')
plt.show()

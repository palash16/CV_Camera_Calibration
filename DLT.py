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
### RANSAC Algorithm for 6 Points
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
    return best_projection_matrix

### Main Code Starts here
### Reading Image
I = cv2.imread('Assignment1_Data/IMG_5455.JPG')
plt.imshow(I)
plt.title('Original Image')
plt.show()

### Manually defining Image Points
I_pts  = [[140,285],[908,338],[1655,391],[2450,431],[3223,465],[4070,535],[4945,552],[200,1045],[941,1112],[1682,1159],
           [2456,1232],[3243,1265],[4064,1326],[4878,1392],[661,2100],[1435,2166],[2222,2233],[3043,2306],[3877,2380],
          [4744,2460],[334,2400],[1148,2460],[1995,2547],[2849,2627],[3737,2713],[4658,2794],[834,2807],[1722,2907],[2629,2987]
          ,[3570,3080],[4551,3167],[474,3214],[1408,3314],[2389,3401],[3377,3488],[4437,3614]]
I_pts = np.asarray(I_pts)

### Definging World Points
X = [216,180,144,108,72,36,0,216,180,144,108,72,36,0,180,144,108,72,36,0,180,144,108,72,36,0,144,108,72,36,0,144,108,72,36,0]
Y = [72,72,72,72,72,72,72,36,36,36,36,36,36,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Z = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,36,36,36,36,36,72,72,72,72,72,72,108,108,108,108,108,144,144,144,144,144]
n = len(I_pts)
I1 = I.copy()
for pt in I_pts:
    I1[pt[1]-25:pt[1]+25,pt[0]-25:pt[0]+25] = [255,0,0] 
plt.imshow(I1)
plt.title('Original Image with marked points')
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
    reprojected_image[pt[1]-25:pt[1]+25,pt[0]-25:pt[0]+25,:] = [0,0,255]
for pt in ppp:
    reprojected_image[pt[1]-25:pt[1]+25,pt[0]-25:pt[0]+25,:] = [255,0,0]

plt.imshow(reprojected_image)
plt.title('DLT      Blue: Original Points     Red: Reprojected Points')
plt.show()


R,K,T = DLT_algorithm(P)
print('Camera Matirx is :')
print(K)
print('Rotation Matrix is :')
print(R)
print('Projection Center is:')
print(T)


### Estimating P matrix with RANSAC Algorithm
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
    reprojected_image[pt[1]-25:pt[1]+25,pt[0]-25:pt[0]+25,:] = [0,0,255]
for pt in ppp:
    reprojected_image[pt[1]-25:pt[1]+25,pt[0]-25:pt[0]+25,:] = [255,0,0]
plt.imshow(reprojected_image)
plt.title('RANSAC DLT       Blue: Original Points     Red: Reprojected Points')
plt.show()

### Plotting the Origin Point
I_origin = I.copy()
origin = [0,0,0,1]
origin = np.asarray(origin)
origin_img = np.zeros((3,1))
origin_img = np.matmul(P_best,origin)
origin_img = origin_img/origin_img[2]
I_origin[int(origin_img[1])-25:int(origin_img[1])+25,int(origin_img[0])-25:int(origin_img[0])+25,:] = [0,255,0]
plt.imshow(I_origin)
plt.title('Green Marker indicates the position of World Point given the Calibration Matrix')
plt.show()


### Plotting the Wireframe on the grid
wireframe = I.copy()
plt.imshow(wireframe)

idx = [6,13,19,25,30,35]
q = 0
for i in range(35):
    if i == idx[q]:
        q = q + 1
        continue
    plt.plot([ppp[i][0],ppp[i+1][0]],[ppp[i][1],ppp[i+1][1]],'go-')
plt.plot([ppp[0][0],ppp[6][0]],[ppp[0][1],ppp[6][1]],'go-')
order = [0,7,-1,1,8,14,20,-1,2,9,15,21,26,31,-1,3,10,16,22,27,32,-1,4,11,17,23,28,33,-1,5,12,18,24,29,34,-1,6,13,19,25,30,35]
for i in range(len(order)-1):
	if (order[i+1] == -1 or order[i] == -1):
		continue
	plt.plot([ppp[order[i]][0],ppp[order[i+1]][0]],[ppp[order[i]][1],ppp[order[i+1]][1]],'go-')
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


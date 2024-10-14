#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Homography, Fundamental Matrix and Two View SfM
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 2.0
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2



# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)



def P(T,K):
    canonical_mat= np.identity(3)
    canonical_mat= np.hstack((canonical_mat,np.array([0,0,0]).reshape(3,1)))
    return K@canonical_mat@T

def pair_equations_camera(x,y,p):
        A= p[2,0]*x-p[0,0]
        B =p[2,1]*x-p[0,1]
        C= p[2,2]*x-p[0,2]
        D= p[2,3]*x-p[0,3]
        E= p[2,0]*y-p[1,0]
        F= p[2,1]*y-p[1,1]
        G= p[2,2]*y-p[1,2]
        H= p[2,3]*y-p[1,3]
        line1= np.array([A,B,C,D])
        line2= np.array([E,F,G,H])
        return np.vstack((line1,line2))


def triangulation(p_1,x1,p_2,x2):
    out= np.ones((x1.shape[1],3))
    for i in range(x1.shape[1]):
        #camera 1 equation
        x= x1[0,i]
        y= x1[1,i]
        eqs_c1= pair_equations_camera(x,y,p_1)
        #camera 2 equations
        x= x2[0,i]
        y= x2[1,i]
        eq_c2= pair_equations_camera(x,y,p_2)
        #compute matrix A
        A= np.vstack((eqs_c1,eq_c2))
        #SVD 
        U, S, Vh= np.linalg.svd(A)
        x_3d=Vh[-1]
        x_3d=x_3d/x_3d[-1]
        out[i]=x_3d[:-1]
    return out

def plot_epipolar_line(f_matrix,x_1,image_2,ax,colr,x_end=1000):
    line= f_matrix@x_1
    x_axis_values= np.arange(0,x_end,1)
    y= -(x_axis_values*line[0]+line[2])/line[1]
    ax.plot(x_axis_values,y,color=colr)
    
def fundamental_matrix(T_w_c1,T_w_c2,K):
    T=T_w_c2@np.linalg.inv(T_w_c1)
    #compute epipolar 
    R= T[:3,:3]
    t=T[:-1,3]
    t_=np.zeros((3,3))
    t_[0,1]=-t[2]
    t_[0,2]=t[1]
    t_[1,0]=t[2]
    t_[1,2]=-t[0]
    t_[2,0]= -t[1]
    t_[2,1]= t[0]
    E=t_@R
    #fundamental matrix
    F=np.linalg.inv(K).T@E@np.linalg.inv(K)
    return F
def rotation(U,V,W):
    R1=U@W@V 
    R2=U@W.T@V
    
    if np.linalg.det(R1)<0:
        R1=-R1
    if np.linalg.det(R2)<0:
        R2=-R2
    return R1,R2

def check_points_front_camera(T_21,x_c1):
    ct=0
    for i in range(x_c1.shape[0]):
        x_c2= T_21@np.append(x_c1[i],[1])
        if(x_c1[i,2]>0 and x_c2[2]>0):
            ct+=1
            
    return ct 

#average distance between ground truth and  generated points    
def distance_pts(x_1,x_2):
    # Compute the pairwise distances between all points in set1 and set2
    distances = np.linalg.norm(x_1[:, np.newaxis] - x_2, axis=2)
    # Calculate the average distance
    average_distance = np.mean(distances)
    return average_distance
   
#compute homography matrix

def homography_matrix(x1,x2):
    A= np.ones((2*x1.shape[1],9))
    row=0
    for i in range(x1.shape[1]):
        x_1= x1[0,i]
        y_1= x1[1,i]
        x_2= x2[0,i]
        y_2= x2[1,i]
        line1=[x_1,y_1,1,0,0,0,-x_1*x_2,-x_2*y_1,-x_2]
        line2=[0,0,0,x_1,y_1,1,-x_1*y_2,-y_2*y_1,-y_2]    
        A[row]=np.array(line1)
        row+=1
        A[row]=np.array(line2)
        row+=1
        
    u,S,Vt= np.linalg.svd(A)
    H= Vt[-1]
    H=H.reshape((3,3))
    return H

#compute point transfer 
def point_transfer(H,x1):
    pts=np.zeros((2,x1.shape[1]))
    for i in range(x1.shape[1]):
        pt=H@x1[:,i]
        pts[:,i]=pt[:2]/pt[-1]
    return pts

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt('data/T_w_c1.txt')
    T_w_c2 = np.loadtxt('data/T_w_c2.txt')
    #3d points ground truth
    X_w = np.loadtxt('data/X_w.txt')
    #camera calibration
    K_c = np.loadtxt('data/K_c.txt')
    #camera 1 image points
    x1 = np.loadtxt('data/x1Data.txt')
    #camera 2 image points
    x2 = np.loadtxt('data/x2Data.txt')
    #image1 points floor 
    x1floor= np.loadtxt("data/x1FloorData.txt")
    #image2 points floor 
    x2floor= np.loadtxt("data/x2FloorData.txt")
    
    #########################################
    ############## Triangulation ############
    #########################################
    
    #compute camera 1 projection matrix
    p_1= P(np.linalg.inv(T_w_c1),K_c)
    #compute camera 2 projection matrix
    p_2= P(np.linalg.inv(T_w_c2),K_c)
    #get 3d world points
    pts_triangulation= triangulation(p_1,x1,p_2,x2)    
    ##Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.',  label="Ground Truth")
    ax.scatter(pts_triangulation[:,0], pts_triangulation[:,1], pts_triangulation[:,2], marker='+', label="Our Sol")
    ax.legend()
    #plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()   
    
    ##############################################################
    #########Funddamental matrix and Structure from Motion########
    ##############################################################
    
    ####################################
    #Fundamental matrix definition(2.1)#
    ####################################
    #Loading test fundamental matrix between image 2 and 1
    f_2_1_test= np.loadtxt("data/F_21_test.txt")
    
    
    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread('imgs/image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('imgs/image2.png'), cv2.COLOR_BGR2RGB)
   
    #epipolar line test plot
    plt.figure(1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    pt1= np.array([x1[0,0],x1[1,0],1.])
    pt2= np.array([x1[0,1],x1[1,1],1.])
    pt3= np.array([x1[0,2],x1[1,2],1.])
    pt4= np.array([x1[0,10],x1[1,0],1.])
    plt.plot(pt1[0], pt1[1],'rx', markersize=10, color="r")
    plt.plot(pt2[0], pt2[1],'rx', markersize=10, color="b")
    plt.plot(pt3[0], pt3[1],'rx', markersize=10, color="orange")
    plt.plot(pt4[0], pt4[1],'rx', markersize=10, color= "g")
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    plt.waitforbuttonpress()
    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plot_epipolar_line(f_2_1_test,pt1,img2,plt,"r")
    plot_epipolar_line(f_2_1_test,pt2,img2,plt,"b")
    plot_epipolar_line(f_2_1_test,pt3,img2,plt,"orange")
    plot_epipolar_line(f_2_1_test,pt4,img2,plt,"g")
    
    plt.title('Epipolar lines')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    ####################################
    #Fundamental matrix definition(2.2)#
    ####################################
    F=fundamental_matrix(T_w_c1,T_w_c2,K_c)
    plt.figure(3)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plot_epipolar_line(F,pt1,img2,plt,"r")
    plot_epipolar_line(F,pt2,img2,plt,"b")
    plot_epipolar_line(F,pt3,img2,plt,"orange")
    plot_epipolar_line(F,pt4,img2,plt,"g")
    plt.title('Fundamental Matrix definition')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    ####################################
    #Fundamental matrix definition(2.3)#
    ####################################
    A= np.zeros((x1.shape[1],9))
    for i in range(x1.shape[1]):
        x1_=np.append(x1[:,i],[1])
        x2_=np.append(x2[:,i],[1])
        b= np.zeros((9,1))
        b[0]=x1_[0]*x2_[0]
        b[1]=x1_[1]*x2_[0]
        b[2]=x1_[2]*x2_[0]
        b[3]=x1_[0]*x2_[1]
        b[4]=x1_[1]*x2_[1]
        b[5]= x1_[2]*x2_[1] 
        b[6]=x1_[0]*x2_[2]
        b[7]=x1_[1]*x2_[2]
        b[8]=x1_[2]*x2_[2]
        A[i,:]= b.T[0]  
    U,D,Vh= np.linalg.svd(A)
    F=Vh[-1].reshape(3,3)
    U_,S_,Vt_=np.linalg.svd(F)
    S__=np.zeros((3,3))
    S__[0,0]=S_[0]
    S__[1,1]=S_[1]
    F_=U_@S__@Vt_
    #Compute epipole 
    U__, DD, VTT= np.linalg.svd(F_.T)
    e= VTT[-1]
    e/=e[-1]
    #plot epipole and epipolar lines
    plt.figure(4)
    pt1= np.array([x1[0,0],x1[1,0],1.])
    pt2= np.array([x1[0,1],x1[1,1],1.])
    pt3= np.array([x1[0,20],x1[1,20],1.])
    pt4= np.array([x1[0,10],x1[1,0],1.])
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plot_epipolar_line(F_,pt1,img2,plt,"r",e[0])
    plot_epipolar_line(F_,pt2,img2,plt,"b",e[0])
    plot_epipolar_line(F_,pt3,img2,plt,"orange",e[0])
    plot_epipolar_line(F_,pt4,img2,plt,"g",e[0])
    
    plt.title('Fundamental Matrix 8 point sol')
    plt.scatter(e[0],e[1])
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    ####################################
    #Fundamental matrix definition(2.4)#
    ####################################
    #compute epipolar matrix
    E_2_1=K_c.T@F_@K_c  
    #compute 4 potential transforms from Epipolar matrix
    U,S,V= np.linalg.svd(E_2_1)
    W=np.zeros((3,3))
    W[0,1]=-1
    W[1,0]=1
    W[2,2]=1
    R_90, R_90_min=rotation(U,V,W)
    #solution 1
    t=U[:,-1]
    T_21_1=ensamble_T(R_90,t)
    #solution 2
    T_21_2=ensamble_T(R_90,-t)
    #solution 3
    T_21_3=ensamble_T(R_90_min,t)
    #solution 4
    T_21_4=ensamble_T(R_90_min,-t)  
    #projection matrix 1    
    T_1= np.zeros((4,4))
    T_1[0,0]=1
    T_1[1,1]=1
    T_1[2,2]=1
    T_1[3,3]=1
    print(T_1)
    P_1= P(T_1,K_c)
    #projection matrices 2
    P2= P(T_21_1,K_c)
    sol1_3dpoints=triangulation(P_1,x1,P2,x2)
    print("SOL1 Nb of points",check_points_front_camera(T_21_1,sol1_3dpoints))
    P2= P(T_21_2,K_c)
    sol2_3dpoints=triangulation(P_1,x1,P2,x2)
    print("SOL2 Nb of points",check_points_front_camera(T_21_2,sol2_3dpoints))
    P2= P(T_21_3,K_c)
    sol3_3dpoints=triangulation(P_1,x1,P2,x2)
    print("SOL3 Nb of points",check_points_front_camera(T_21_3,sol3_3dpoints))
    P2= P(T_21_4,K_c)
    sol4_3dpoints=triangulation(P_1,x1,P2,x2)
    print("SOL4 Nb of points",check_points_front_camera(T_21_4,sol4_3dpoints))
    print("Exercise 2.4")
    print("Rotation Camera Pose SOlution",R_90_min)    
    print("Translation Camera Pose SOlution",t)    
    ####################################
    #Fundamental matrix definition(2.5)#
    ####################################
    print("Exercise 2.5")
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')
    pts_sol =np.zeros((sol3_3dpoints.shape[0],sol3_3dpoints.shape[1]))
    
    for i in range(pts_sol.shape[0]):
        pt=np.append(sol3_3dpoints[i],1)
        pt=T_w_c1@pt
        pts_sol[i]=pt[:3]
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.',  label="Ground Truth")
    ax.scatter(pts_sol[:,0], pts_sol[:,1], pts_sol[:,2], marker='+', label="Our Sol")
    ax.legend()
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.draw()   
    plt.waitforbuttonpress()
    print("Average distance between ground truth and our pts:",distance_pts(X_w.T[:,:3],pts_sol))
    
    
    ####################################
    ##############Homographies(3)#######
    ####################################
    
    ####################################
    #Homographies(3.3)##################
    ####################################
    #homography_matrix
    H=homography_matrix(x1floor,x2floor)
    print("Exercise 3.3")
    print("H=",H)    
    ####################################
    #Homographies(3.1)##################
    ####################################
    d=1.7257
    n=np.array([0.0149,0.9483,0.3171])
    H=K_c@(R_90_min-t@n.T*(1./d))@np.linalg.inv(K_c)
    print("Exercise 3.1")
    print("H=",H)
    ####################################
    #Homographies(3.2)##################
    ####################################
    x1_=point_transfer(H,x1floor)
    # feature points
    plt.figure(5)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x1, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    plt.figure(6)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0, :], x2[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x2, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plotNumberedImagePoints(x1_, 'g', (10,0))
    plt.plot(x1_[0, :], x1_[1, :],'gx', markersize=10)
    plt.title('Image 2')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
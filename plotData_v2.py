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
        U, S, Vh= np.linalg.svd(A,full_matrices=True)
        x_3d=Vh[-1]
        x_3d=x_3d/x_3d[-1]
        out[i]=x_3d[:-1]
    return out

def plot_epipolar_line(f_matrix,x_1,image_2,ax,colr):
    line= f_matrix@x_1
    x_axis_values= np.arange(0,image_2.shape[1],1)
    y= -(x_axis_values*line[0]+line[2])/line[0]
    ax.plot(x_axis_values,y,color=colr)
    
def fundamental_matrix(T_w_c1,T_w_c2,K):
    T=np.linalg.inv(T_w_c1)@T_w_c2
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
    F=K.T@E@np.linalg.inv(K)
    return F

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
    
    #Fundamental matrix definition
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
    
    
    # feature points
    plt.figure(4)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x1, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    plt.figure(5)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0, :], x2[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x2, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 2')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
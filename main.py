import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import argparse

from ransac_ import calc_error, ransac
from q1 import find_camera_and_3dpts
from triangulate_ import triangulate

def normalize(v):
    # return v / np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return v / np.sqrt(np.sum(v**2))

def getP(pts2d, pts3d):

    pts2d_homo = np.hstack((pts2d, np.ones((pts2d.shape[0],1))))

    A = []
    for i, pt_2d in enumerate(pts2d_homo):

        x_2d = normalize(pt_2d)
        x_3d = normalize(pts3d[i])

        eq1 = np.hstack((np.zeros(4,), -x_2d[2]*x_3d, x_2d[1]*x_3d))
        eq2 = np.hstack((x_2d[2]*x_3d, np.zeros(4,), -x_2d[0]*x_3d))

        A.append(eq1)
        A.append(eq2)

    A = np.array(A)
    u, diag, v = np.linalg.svd(A)

    P = v[-1,:].reshape(3,4)
    print("projection matrix")
    print(P)
    return P


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=str, help = "question number")
    args = parser.parse_args()


    if args.question=='q1':
        intrinsics = np.load('./data/monument/intrinsics.npy', allow_pickle=True)
        corresp = np.load('./data/monument/some_corresp_noisy.npz')

        print(intrinsics)
        print(type(corresp))
        # print(corresp['pts1'].shape,corresp['pts2'].shape)

        K1 = np.array([[1.07497032e+03, 0.00000000e+00, 3.89500000e+02],
                    [0.00000000e+00, 1.07497032e+03, 5.26000000e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        K2 = np.array([[2.72135984e+03, 0.00000000e+00, 3.37500000e+02],
                    [0.00000000e+00, 2.72135984e+03, 5.06000000e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # K1 = intrinsics['K1']
        # K2 = intrinsics['K2']

        print(K1)
        print(K2)

        pts1 = corresp['pts1']
        pts2 = corresp['pts2']

        F, iters, percent_inliers, pts1_inliers, pts2_inliers = ransac(pts1, pts2)

        # error = calc_error(pts1,pts2, F)
        # print(np.mean(error))
        # print(np.sum(error<5))

        # fig, ax = plt.subplots()
        # ax.plot(iters, percent_inliers)
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Percent Inliers')
        # plt.show()

        pts3d, (R, C) = find_camera_and_3dpts(pts1_inliers, pts2_inliers, K1, K2)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])
        plt.show()

        print(R)
        print(C)



    elif args.question=='q2':
        cam1 = np.load('./data/data_cow/cameras/cam1.npz')
        cam2 = np.load('./data/data_cow/cameras/cam2.npz')

        K = cam1['K']
        print(K)

        pts1 = np.load('./data/data_cow/correspondences/pairs_1_2/cam1_corresp.npy')
        pts2 = np.load('./data/data_cow/correspondences/pairs_1_2/cam2_corresp.npy')

        print(pts1.shape)
        print(pts2.shape)

        pts3d, (R, C) = find_camera_and_3dpts(pts1, pts2, K, K)
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])
        # plt.show()

        print(R)
        print(C)

        P1 = np.eye(3)
        P1 = K.dot(np.hstack((P1, np.zeros((3,1)))))
        P2 = K.dot(np.hstack((R, C)))


        # Register camera 3
        cam1_pts = np.load('./data/data_cow/correspondences/pairs_1_3/cam1_corresp.npy')
        cam3_pts = np.load('./data/data_cow/correspondences/pairs_1_3/cam2_corresp.npy')

        print(cam1_pts.shape)

        indices = np.array([], dtype=np.int)
        ind2 = []
        for i,row in enumerate(cam1_pts):
            tmp = np.where((pts1 == row).all(axis=1))[0]
            indices = np.hstack((indices,tmp))
            if tmp.shape[0]!=0:
                ind2.append(i)

        # print(indices.shape, len(ind2))
        # for i, idx in enumerate(indices):
        #     print(pts1[idx], cam1_pts[ind2[i]])
 
        P3 = getP(cam3_pts[ind2], pts3d[indices])

    
        # Triangulate points between camera 1 and 3
        cam3_pts3d_1 = triangulate(cam1_pts, cam3_pts, P1, P3)
        cam3_pts3d_1 = cam3_pts3d_1/cam3_pts3d_1[:,-1].reshape(-1,1)

        # Triangulate points between camera 2 and 3
        cam2_pts = np.load('./data/data_cow/correspondences/pairs_2_3/cam1_corresp.npy')
        cam3_pts = np.load('./data/data_cow/correspondences/pairs_2_3/cam2_corresp.npy')
        cam3_pts3d_2 = triangulate(cam2_pts, cam3_pts, P2, P3)
        cam3_pts3d_2 = cam3_pts3d_2/cam3_pts3d_2[:,-1].reshape(-1,1)

        
        # Register camera 4
        cam1_pts = np.load('./data/data_cow/correspondences/pairs_1_4/cam1_corresp.npy')
        cam4_pts = np.load('./data/data_cow/correspondences/pairs_1_4/cam2_corresp.npy')

        indices = np.array([], dtype=np.int)
        ind2 = []
        for i,row in enumerate(cam1_pts):
            tmp = np.where((pts1 == row).all(axis=1))[0]
            indices = np.hstack((indices,tmp))
            if tmp.shape[0]!=0:
                ind2.append(i)

        # print(indices.shape, len(ind2))
        # for i, idx in enumerate(indices):
        #     print(pts1[idx], cam1_pts[ind2[i]])

        P4 = getP(cam4_pts[ind2], pts3d[indices])

        # Triangulate points between camera 1 and 4
        cam4_pts3d_1 = triangulate(cam1_pts, cam4_pts, P1, P4)
        cam4_pts3d_1 = cam4_pts3d_1/cam4_pts3d_1[:,-1].reshape(-1,1)

        # Triangulate points between camera 2 and 4
        cam2_pts = np.load('./data/data_cow/correspondences/pairs_2_4/cam1_corresp.npy')
        cam4_pts = np.load('./data/data_cow/correspondences/pairs_2_4/cam2_corresp.npy')
        cam4_pts3d_2 = triangulate(cam2_pts, cam4_pts, P2, P4)
        cam4_pts3d_2 = cam4_pts3d_2/cam4_pts3d_2[:,-1].reshape(-1,1)

        # Triangulate points between camera 3 and 4
        cam3_pts = np.load('./data/data_cow/correspondences/pairs_3_4/cam1_corresp.npy')
        cam4_pts = np.load('./data/data_cow/correspondences/pairs_3_4/cam2_corresp.npy')
        cam4_pts3d_3 = triangulate(cam3_pts, cam4_pts, P3, P4)
        cam4_pts3d_3 = cam4_pts3d_3/cam4_pts3d_3[:,-1].reshape(-1,1)


        # Plot
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], c=pts3d[:,1])
        # ax.scatter(cam3_pts3d_1[:,0], cam3_pts3d_1[:,1], cam3_pts3d_1[:,2], c=cam3_pts3d_1[:,1])
        # ax.scatter(cam3_pts3d_2[:,0], cam3_pts3d_2[:,1], cam3_pts3d_2[:,2], c=cam3_pts3d_2[:,1])

        # ax.scatter(cam4_pts3d_1[:,0], cam4_pts3d_1[:,1], cam4_pts3d_1[:,2], c=cam4_pts3d_1[:,1])
        # ax.scatter(cam4_pts3d_2[:,0], cam4_pts3d_2[:,1], cam4_pts3d_2[:,2], c=cam4_pts3d_2[:,1])
        # ax.scatter(cam4_pts3d_3[:,0], cam4_pts3d_3[:,1], cam4_pts3d_3[:,2], c=cam4_pts3d_3[:,1])

        # plt.show()


        #Open3d viz
        xyz = pts3d[:,:3]

        # xyz = np.vstack((xyz, cam3_pts3d_1[:,:3]))
        # xyz = np.vstack((xyz, cam3_pts3d_2[:,:3]))

        # xyz = np.vstack((xyz, cam4_pts3d_1[:,:3]))
        # xyz = np.vstack((xyz, cam4_pts3d_2[:,:3]))
        # xyz = np.vstack((xyz, cam4_pts3d_3[:,:3]))
       
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.visualization.draw_geometries([pcd])


        Kinv = np.linalg.inv(K)
        # R = Kinv.dot(P[:,:-1])
        # C = Kinv.dot(P[:,-1:])
        # print(R)
        # print(C)

        RT = Kinv.dot(P3)
        print(RT)

        u, diag, v = np.linalg.svd(RT[:,:-1])
        print(u.dot(v))

        RT = Kinv.dot(P4)
        print(RT)

        u, diag, v = np.linalg.svd(RT[:,:-1])
        print(u.dot(v))
import numpy as np

def triangulate(pts1, pts2, P1, P2):

    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0],1))))

    pts_3d = []
    for x1, x2 in zip(pts1_homo, pts2_homo):

        A = []
        A.append(x1[0]*P1[2] - P1[0])
        A.append(x1[1]*P1[2] - P1[1])
        A.append(x2[0]*P2[2] - P2[0])
        A.append(x2[1]*P2[2] - P2[1])

        u, diag, v = np.linalg.svd(A)

        X = v[-1,:]
        pts_3d.append(X)

    return np.array(pts_3d)



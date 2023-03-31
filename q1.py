import numpy as np
from triangulate_ import triangulate
from eightPt import eightPt_algo

def find_camera_and_3dpts(pts1, pts2, K1, K2):

    F = eightPt_algo(pts1, pts2)
    E = K2.T.dot(F.dot(K1))

    u, diag, v = np.linalg.svd(E)
    diag = np.diag([1,1,0])
    E = u.dot(diag.dot(v))
    # print(E)

    W = np.zeros((3,3))
    W[0,1] = -1
    W[1,0] = W[2,2] = 1

    R1 = u.dot(W.dot(v))
    R2 = u.dot((W.T).dot(v))

    C1 = u[:,-1].reshape(-1,1)
    C2 = -C1

    P1 = np.eye(3)
    P1 = K1.dot(np.hstack((P1, np.zeros((3,1)))))

    P2_1 = K2.dot(np.hstack((R1, C1)))
    P2_2 = K2.dot(np.hstack((R1, C2)))
    P2_3 = K2.dot(np.hstack((R2, C1)))
    P2_4 = K2.dot(np.hstack((R2, C2)))

    cam2_RC = [(R1,C1), (R1,C2), (R2,C1), (R2,C2)]
    cam_front_pts = -np.inf
    best_RC = None
    best_3dpts = None

    for R,C in cam2_RC:

        P2 = K2.dot(np.hstack((R, C)))
        
        pts3d = triangulate(pts1, pts2, P1, P2)
        pts3d = pts3d/pts3d[:,-1].reshape(-1,1)

        tmp = (pts3d[:,:-1] - C.reshape(3,)).dot(R[-1,:].reshape(-1,1))

        if np.sum(tmp>0) > cam_front_pts:
            cam_front_pts = np.sum(tmp>0)
            best_RC = (R,C)
            best_3dpts = pts3d

    return best_3dpts, best_RC
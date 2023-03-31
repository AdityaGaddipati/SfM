import numpy as np

def calc_tf(pts):
    x0, y0 = pts.mean(axis=0)
    s = np.sqrt(2)/(np.sqrt(((pts - [x0,y0])**2).sum(axis=1)).mean())
    print(s)
    T = np.diag([1.0,1.0,1.0])
    T[0,0] = T[1,1] = s
    T[0,2] = -s*x0
    T[1,2] = -s*y0
    print(T)
    return T


def eightPt_algo(x_, x_dash):

    T = calc_tf(x_)
    x_ = np.hstack((x_, np.ones((x_.shape[0],1))))
    x_ = (T.dot(x_.T)).T

    T_dash = calc_tf(x_dash)
    x_dash = np.hstack((x_dash, np.ones((x_dash.shape[0],1))))
    x_dash = (T_dash.dot(x_dash.T)).T

    A = np.hstack((x_dash[:,0].reshape(-1,1)*x_ , x_dash[:,1].reshape(-1,1)*x_, x_dash[:,2].reshape(-1,1)*x_))
    # print(A.shape)

    u, diag, v = np.linalg.svd(A)
    F_hat = v[-1,:].reshape(3,3)
    # print(F_hat)

    u, diag, v = np.linalg.svd(F_hat)
    # print(diag)

    diag[-1] = 0
    F_hat = u.dot(np.diag(diag).dot(v))
    # print(F_hat)

    F = (T_dash.T).dot(F_hat.dot(T))
    print(F)

    return F/F[2,2]
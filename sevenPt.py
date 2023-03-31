import numpy as np

def singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F

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

def sevenPt_algo(x_, x_dash):

    T = calc_tf(x_)
    x_ = np.hstack((x_, np.ones((x_.shape[0],1))))
    x_ = (T.dot(x_.T)).T

    T_dash = calc_tf(x_dash)
    x_dash = np.hstack((x_dash, np.ones((x_dash.shape[0],1))))
    x_dash = (T_dash.dot(x_dash.T)).T

    A = np.hstack((x_dash[:,0].reshape(-1,1)*x_ , x_dash[:,1].reshape(-1,1)*x_, x_dash[:,2].reshape(-1,1)*x_))
    # print(A.shape)

    u, diag, v = np.linalg.svd(A)

    f1 = v[-1,:].reshape(3,3)
    f2 = v[-2,:].reshape(3,3)

    f = np.stack((f1,f2),0)

    D = np.zeros((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                tmp = np.zeros((3,3))
                tmp[:,0] = f[i,:,0]
                tmp[:,1] = f[j,:,1]
                tmp[:,2] = f[k,:,2]
                D[i,j,k] = np.linalg.det(tmp)

    coefs = np.zeros(4,)
    coefs[0] = -D[1][0][0]+D[0][1][1]+D[0][0][0]+D[1][1][0]+D[1][0][1]-D[0][1][0]-D[0][0][1]-D[1][1][1]
    coefs[1] = D[0][0][1]-2*D[0][1][1]-2*D[1][0][1]+D[1][0][0]-2*D[1][1][0]+D[0][1][0]+3*D[1][1][1]
    coefs[2] = D[1][1][0]+D[0][1][1]+D[1][0][1]-3*D[1][1][1]
    coefs[3] = D[1][1][1]
    
    roots = np.polynomial.polynomial.polyroots(coefs)
    # print(roots)

    F_mats = []
    for a in roots:
        if np.isreal(a):
            F = a*f1 + (1-a)*f2
            F = singularize(F)
            F = (T_dash.T).dot(F.dot(T))
            F = F/F[2,2]
            F_mats.append(F.real)

    return F_mats
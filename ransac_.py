import numpy as np
from eightPt import eightPt_algo
from sevenPt import sevenPt_algo


def calc_error(pts1, pts2, F):
    
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0],1))))

    line1s = pts1_homo.dot(F.T)
    dist1 = np.square(np.divide(np.sum(np.multiply(
        line1s, pts2_homo), axis=1), np.linalg.norm(line1s[:, :2], axis=1)))

    line2s = pts2_homo.dot(F)
    dist2 = np.square(np.divide(np.sum(np.multiply(
        line2s, pts1_homo), axis=1), np.linalg.norm(line2s[:, :2], axis=1)))

    error = (dist1 + dist2).flatten()
    return error

def ransac(pts1, pts2):

    N = pts1.shape[0]
    maxIter = 1000
    threshold = 10
    curr_inliers = None
    best_inliers = -np.inf

    percent_inliers = []
    iters = []

    for i in range(maxIter):
        indices = np.random.randint(0, N, (7,))
        Fs = sevenPt_algo(pts1[indices], pts2[indices])

        # indices = np.random.randint(0, N, (8,))
        # Fs = [eightPt_algo(pts1[indices], pts2[indices])]
        
        for F in Fs:
            error = calc_error(pts1, pts2, F)
            inliers = error<threshold
            num_inliers = np.sum(inliers)
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                curr_inliers = inliers

            iters.append(i)
            percent_inliers.append((best_inliers/N)*100)

    F = eightPt_algo(pts1[curr_inliers], pts2[curr_inliers])

    return F, iters, percent_inliers, pts1[curr_inliers], pts2[curr_inliers]
"""SIFT-seeded local Lucas-Kanade refinement in upright full-resolution pixels."""
import cv2
import numpy as np


def refine_matches(gray_left, gray_right, left, right, fb_limit_px=1.0,
                   correction_limit_px=8.0):
    a = np.asarray(left, np.float32).reshape(-1, 1, 2)
    b = np.asarray(right, np.float32).reshape(-1, 1, 2)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Nonempty paired starting points are required")
    if gray_left.shape != gray_right.shape or gray_left.ndim != 2:
        raise ValueError("Grayscale image dimensions must agree")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Coordinates must be finite")
    if fb_limit_px <= 0 or correction_limit_px <= 0:
        raise ValueError("Thresholds must be positive")
    settings = dict(winSize=(31, 31), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, .001),
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4)
    # SIFT provides the large displacement. LK only refines its local neighbourhood.
    refined, status_f, _ = cv2.calcOpticalFlowPyrLK(gray_left, gray_right, a, b.copy(), **settings)
    if refined is None:
        raise ValueError("Forward patch refinement failed")
    finite_forward = np.isfinite(refined).all(axis=(1, 2))
    safe_forward = refined.copy()
    safe_forward[~finite_forward] = b[~finite_forward]
    returned, status_b, _ = cv2.calcOpticalFlowPyrLK(gray_right, gray_left, safe_forward, a.copy(), **settings)
    if returned is None:
        raise ValueError("Backward patch refinement failed")
    fb = np.linalg.norm(returned[:, 0]-a[:, 0], axis=1)
    correction = np.linalg.norm(refined[:, 0]-b[:, 0], axis=1)
    h, w = gray_left.shape
    def inside(points):
        x,y=points[:,0,0],points[:,0,1]
        return (x>=16)&(x<w-16)&(y>=16)&(y<h-16)
    finite = finite_forward & np.isfinite(returned).all(axis=(1,2))
    tracked = status_f.ravel().astype(bool) & status_b.ravel().astype(bool)
    keep = finite & tracked & inside(a) & inside(refined) & inside(returned)
    keep &= (fb<=fb_limit_px) & (correction<=correction_limit_px)
    return dict(left=a[:,0].astype(float), right=refined[:,0].astype(float),
                returned_left=returned[:,0].astype(float), keep=keep,
                forward_backward_px=fb, correction_px=correction,
                forward_status=status_f.ravel().astype(bool), backward_status=status_b.ravel().astype(bool))

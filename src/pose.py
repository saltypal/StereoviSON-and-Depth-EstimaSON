"""Exploratory two-view pose. Coordinates use upright pixels; translation is metres."""
import cv2
import numpy as np


def reconstruct(left, right, camera_matrix, baseline_m, seed=42):
    left, right = np.asarray(left, float), np.asarray(right, float)
    K = np.asarray(camera_matrix, float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2 or len(left) < 8:
        raise ValueError("At least eight paired 2D points are required")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.isfinite(K).all():
        raise ValueError("Non-finite coordinates or intrinsics")
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("Baseline must be positive metres")
    cv2.setRNGSeed(seed)
    E, mask = cv2.findEssentialMat(left, right, K, method=cv2.RANSAC,
                                  prob=0.999, threshold=1.5, maxIters=10000)
    if E is None or E.shape != (3, 3) or mask is None:
        raise ValueError("Essential matrix estimation failed or is ambiguous")
    essential = mask.ravel().astype(bool)
    # Large distance limit avoids recoverPose's default far-point cutoff.
    count, R, t, pose_mask, _ = cv2.recoverPose(E, left, right, K, distanceThresh=1e6, mask=mask.copy())
    pose = pose_mask.ravel().astype(bool)
    if count < 8:
        raise ValueError("Too few points in front of both recovered cameras")
    translation = t.ravel() / np.linalg.norm(t) * baseline_m
    a, b = left[pose], right[pose]
    P1 = K @ np.column_stack((np.eye(3), np.zeros(3)))
    P2 = K @ np.column_stack((R, translation))
    q = cv2.triangulatePoints(P1, P2, a.T, b.T).T
    valid = np.abs(q[:, 3]) > 1e-12
    xyz = np.full((len(q), 3), np.nan)
    xyz[valid] = q[valid, :3] / q[valid, 3:4]
    xyz2 = xyz @ R.T + translation
    valid &= np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (xyz2[:, 2] > 0)
    if valid.sum() < 8:
        raise ValueError("Triangulation failed positive depth checks")
    indices = np.flatnonzero(pose)[valid]
    xyz, xyz2 = xyz[valid], xyz2[valid]
    def project(points):
        pix = points @ K.T
        return pix[:, :2] / pix[:, 2:3]
    err = np.maximum(np.linalg.norm(project(xyz)-left[indices], axis=1),
                     np.linalg.norm(project(xyz2)-right[indices], axis=1))
    ray1 = np.column_stack((left[indices], np.ones(len(indices)))) @ np.linalg.inv(K).T
    ray2 = (np.column_stack((right[indices], np.ones(len(indices)))) @ np.linalg.inv(K).T) @ R
    ray1 /= np.linalg.norm(ray1, axis=1, keepdims=True)
    ray2 /= np.linalg.norm(ray2, axis=1, keepdims=True)
    parallax = np.degrees(np.arccos(np.clip(np.sum(ray1*ray2,axis=1), -1, 1)))
    return dict(E=E, R=R, translation_m=translation, indices=indices, xyz_m=xyz,
                essential_count=int(essential.sum()), positive_count=len(indices),
                cheirality_fraction=float(len(indices)/max(essential.sum(),1)),
                reprojection_px=err, parallax_deg=parallax,
                median_depth_m=float(np.median(xyz[:,2])),
                median_range_m=float(np.median(np.linalg.norm(xyz,axis=1))),
                rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))))

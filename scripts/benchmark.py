import argparse
import h5py
import numpy as np
from scipy.linalg import orthogonal_procrustes


def align_w_scale(mtx1, mtx2, return_trafo=False):
    '''
    mtx1, mtx2: [J,3]

    return aligned mtx2 [J,3]
    '''
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1

    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def compute_pck_with_bbox_np(gt: np.ndarray, pred: np.ndarray, thr: float) -> float:
    """
    Compute PCK (Percentage of Correct Keypoints) with bbox-based normalization using NumPy.

    Args:
        gt (np.ndarray): Ground truth keypoints, shape (N, J, 2)
        pred (np.ndarray): Predicted keypoints, shape (N, J, 2)
        thr (float): Threshold multiplier, final threshold = thr * max(h, w)

    Returns:
        float: PCK score, range [0, 1]
    """
    N, J, _ = gt.shape

    # Step 1: Compute bbox dimensions (h, w) for each sample
    xmin = np.min(gt[..., 0], axis=1)  # (N,)
    xmax = np.max(gt[..., 0], axis=1)  # (N,)
    ymin = np.min(gt[..., 1], axis=1)  # (N,)
    ymax = np.max(gt[..., 1], axis=1)  # (N,)
    w = xmax - xmin  # (N,)
    h = ymax - ymin  # (N,)
    scale = np.maximum(h, w)  # (N,), take max of h and w
    # Step 2: Compute per-sample threshold thr_n = thr * scale
    thr_n = thr * scale  # (N,)
    # Step 3: Compute Euclidean distance for each keypoint, shape (N, J)
    diff = gt - pred
    dist = np.linalg.norm(diff, axis=-1)  # (N, J)
    # Step 4: Determine correct keypoints: dist <= thr_n (thr_n is per-sample)
    # Expand thr_n to (N, J) for broadcasting
    correct = dist <= thr_n[:, np.newaxis]  # (N, J)
    return np.sum(correct) / (N * J)


def main(prediction: str):
    prediction = h5py.File(prediction)

    gt = prediction["joint_cam_gt"][:]
    pred = prediction["joint_cam_pred"][:]
    gt_rel = gt - gt[:, :1]
    pred_rel = pred - pred[:, :1]
    gt_img = prediction["joint_reproj_gt"][:]
    pred_img = prediction["joint_reproj_pred"][:]

    mprpe = np.mean(np.sqrt(np.sum((gt[:, 0] - pred[:, 0]) ** 2, axis=-1)))
    mpjpe_cs = np.mean(np.mean(np.sqrt(np.sum((gt - pred) ** 2, axis=-1)), axis=-1))
    mpjpe_rel = np.mean(np.mean(np.sqrt(np.sum((gt_rel - pred_rel) ** 2, axis=-1)), axis=-1))

    # Calculate PA-aligned metrics
    errors_pa = []
    for ix in range(len(gt)):
        pred_align = align_w_scale(gt[ix], pred[ix])
        errors_pa.append(np.mean(np.sqrt(np.sum((gt[ix] - pred_align) ** 2, axis=-1))).item())
    mpjpe_pa = np.mean(errors_pa)

    # Z error
    error_z = (gt - pred)[..., 2]  # [N,J]
    mean_error_z = np.mean(error_z)
    error_root_z = (gt - pred)[:, 0, 2]  # [N,J]
    mean_error_root_z = np.mean(error_root_z)

    # PCKs
    pck_05 = compute_pck_with_bbox_np(gt_img, pred_img, 0.05)
    pck_10 = compute_pck_with_bbox_np(gt_img, pred_img, 0.10)
    pck_15 = compute_pck_with_bbox_np(gt_img, pred_img, 0.15)

    print(f"mprpe: {mprpe.item()} mm")
    print(f"mpjpe_cs: {mpjpe_cs.item()} mm")
    print(f"mpjpe_rs: {mpjpe_rel.item()} mm")
    print(f"mpjpe_pa: {mpjpe_pa} mm")
    print("")
    print(f"mean_error_z: {mean_error_z}")
    print(f"mean_error_root_z: {mean_error_root_z}")
    print("")
    print(f"pck@0.05: {pck_05}")
    print(f"pck@0.10: {pck_10}")
    print(f"pck@0.15: {pck_15}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate the result")
    parser.add_argument("prediction", type=str, help="prediction result path")
    arg = parser.parse_args()

    main(arg.prediction)
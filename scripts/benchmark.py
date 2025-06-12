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


def main(prediction: str):
    prediction = h5py.File(prediction)

    gt = prediction["joint_cam_gt"][:]
    pred = prediction["joint_cam_pred"][:]
    gt_rel = gt - gt[:, :1]
    pred_rel = pred - pred[:, :1]

    mprpe = np.mean(np.sqrt(np.sum((gt[:, 0] - pred[:, 0]) ** 2, axis=-1)))
    mpjpe_cs = np.mean(np.mean(np.sqrt(np.sum((gt - pred) ** 2, axis=-1)), axis=-1))
    mpjpe_rel = np.mean(np.mean(np.sqrt(np.sum((gt_rel - pred_rel) ** 2, axis=-1)), axis=-1))

    # Calculate PA-aligned metrics
    errors_pa = []
    for ix in range(len(gt)):
        pred_align = align_w_scale(gt[ix], pred[ix])
        errors_pa.append(np.mean(np.sqrt(np.sum((gt[ix] - pred_align) ** 2, axis=-1))).item())
    mpjpe_pa = np.mean(errors_pa)

    print(f"mprpe: {mprpe.item()} mm")
    print(f"mpjpe_cs: {mpjpe_cs.item()} mm")
    print(f"mpjpe_rs: {mpjpe_rel.item()} mm")
    # print(f"mpjpe_pa: {mpjpe_pa.item()} mm")
    print(f"mpjpe_pa: {mpjpe_pa} mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate the result")
    parser.add_argument("prediction", type=str, help="prediction result path")
    arg = parser.parse_args()

    main(arg.prediction)
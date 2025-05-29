import argparse
import h5py
import numpy as np


def main(prediction: str):
    prediction = h5py.File(prediction)

    gt = prediction["joint_cam_gt"][:]
    pred = prediction["joint_cam_pred"][:]
    gt_rel = gt - gt[:, :1]
    pred_rel = pred - pred[:, :1]

    mprpe = np.mean(np.sqrt(np.sum((gt[:, 0] - pred[:, 0]) ** 2, axis=-1)))
    mpjpe_cs = np.mean(np.mean(np.sqrt(np.sum((gt - pred) ** 2, axis=-1)), axis=-1))
    mpjpe_rel = np.mean(np.mean(np.sqrt(np.sum((gt_rel - pred_rel) ** 2, axis=-1)), axis=-1))

    print(f"mprpe: {mprpe.item()} mm")
    print(f"mpjpe_cs: {mpjpe_cs.item()} mm")
    print(f"mpjpe_rs: {mpjpe_rel.item()} mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate the result")
    parser.add_argument("prediction", type=str, help="prediction result path")
    arg = parser.parse_args()

    main(arg.prediction)
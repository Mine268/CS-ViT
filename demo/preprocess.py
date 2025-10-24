# Preprocess the viedo files and extract hands bboxes from it
import cv2
import numpy as np
import argparse

import mediapipe as mp


def main(video_file: str):
    video = cv2.VideoCapture(video_file)

    if not video.isOpened():
        raise FileNotFoundError(f"cannot open {video_file}")

    # extract every frame and detects for right hand


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)

    args = parser.parse_args()
    main(args.file)
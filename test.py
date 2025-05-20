import os
import cv2
import numpy as np
from os import path as osp
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.train import parse_options
from basicsr.data import build_dataloader, build_dataset
from pprint import pprint
from tqdm import tqdm


def test_dataloader():
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt = parse_options(root_path, is_train=False)
    pprint(opt)

    # opt["datasets"]["train"][
    #     "dataroot_gt"
    # ] = "/data/zengzihua/datasets/face_related/p2p_smile_fireflow_ori"
    train_set = build_dataset(opt["datasets"]["train"])

    # img_in img_large 的 shape 都为 [3, 512, 512]
    for item in train_set:
        img_in = item["in"]
        img_large = item["in_large_de"]
        # Convert tensors to numpy arrays for saving
        img_in_np = img_in.permute(1, 2, 0).numpy()  # CHW to HWC
        img_large_np = img_large.permute(1, 2, 0).numpy()  # CHW to HWC

        # Denormalize if images were normalized (assuming mean=0.5, std=0.5)
        img_in_np = (img_in_np * 0.5 + 0.5) * 255.0
        img_large_np = (img_large_np * 0.5 + 0.5) * 255.0

        # Convert from RGB to BGR for OpenCV
        img_in_np = cv2.cvtColor(img_in_np, cv2.COLOR_RGB2BGR)
        img_large_np = cv2.cvtColor(img_large_np, cv2.COLOR_RGB2BGR)

        # Ensure pixel values are in the correct range
        img_in_np = np.clip(img_in_np, 0, 255).astype(np.uint8)
        img_large_np = np.clip(img_large_np, 0, 255).astype(np.uint8)

        # Create concatenated image (horizontally)
        concat_img = np.concatenate((img_in_np, img_large_np), axis=1)

        # Create output directory if it doesn't exist
        os.makedirs("debug_images", exist_ok=True)

        # Save concatenated image
        cv2.imwrite("debug_images/concat_img.png", concat_img)

        print(f"Concatenated image saved to debug_images folder")

        break
        pass


def split_dataset():
    root_path = "/data/zengzihua/datasets/face_related/face_enhencer_1123/val"
    output_ori = "/data/zengzihua/datasets/face_related/face_enhencer_1123/val_inp"
    output_smile = "/data/zengzihua/datasets/face_related/face_enhencer_1123/val_gt"

    os.makedirs(output_ori, exist_ok=True)
    os.makedirs(output_smile, exist_ok=True)

    for fns in tqdm(os.listdir(root_path)):
        img = cv2.imread(osp.join(root_path, fns))
        if img is None:
            continue
        h, w, _ = img.shape
        img_ori = img[:, : w // 2, :]
        img_smile = img[:, w // 2 :, :]
        cv2.imwrite(osp.join(output_ori, fns), img_ori)
        cv2.imwrite(osp.join(output_smile, fns), img_smile)


def test_model():
    pass


if __name__ == "__main__":
    # split_dataset()
    test_dataloader()
    pass

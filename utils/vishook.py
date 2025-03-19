from mmengine.hooks import Hook
import shutil
import cv2
import os.path as osp
import torch
import numpy as np
import os
class SegVisHook(Hook):

    def __init__(self, data_root, vis_num=1) -> None:
        super().__init__()
        self.vis_num = vis_num
        self.palette = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 255, 0): 2,
            (255, 255, 0): 3,
            (0, 0, 255): 4,
            (255, 0, 255): 5,
            (0, 0, 124): 6,
            (202, 202, 202): 7,
            (255, 255, 255): 8,
            (124, 0, 0): 9,
            (0, 124, 0): 10,
            (124, 124, 0): 11,
            (0, 124, 124): 12,
            (0, 255, 255): 13,
            (0, 200, 0): 14,
            (50, 255, 0): 15,
            (50, 124, 0): 16,
            (124, 255, 0): 17,
            (50, 200, 0): 18,
            (0, 255, 50): 19,
            (0, 124, 255): 20,
            (124, 0, 255): 21,
            (50, 50, 255): 22
        }

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch=None,
                       outputs=None) -> None:
        if batch_idx > self.vis_num:
            return
        preds, data_samples = outputs
        img_paths = data_samples['img_path']
        mask_paths = data_samples['mask_path']
        _, C, H, W = preds.shape
        preds = torch.argmax(preds, dim=1)
        for idx, (pred, img_path,
                  mask_path) in enumerate(zip(preds, img_paths, mask_paths)):
            pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
            runner.visualizer.set_image(pred_mask)
            for color, class_id in self.palette.items():
                runner.visualizer.draw_binary_masks(
                    pred == class_id,
                    colors=[color],
                    alphas=1.0,
                )
            # Convert RGB to BGR
            pred_mask = runner.visualizer.get_image()[..., ::-1]
            saved_dir = osp.join(runner.log_dir, 'vis_data', str(idx))
            os.makedirs(saved_dir, exist_ok=True)

            shutil.copyfile(img_path,
                            osp.join(saved_dir, osp.basename(img_path)))
            shutil.copyfile(mask_path,
                            osp.join(saved_dir, osp.basename(mask_path)))
            cv2.imwrite(
                osp.join(saved_dir, f'pred_{osp.basename(img_path)}'),
                pred_mask)
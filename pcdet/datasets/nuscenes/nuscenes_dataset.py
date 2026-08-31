import json
import sys

import yaml
import argparse
from easydict import EasyDict
import copy
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image


NUSCENES_EVAL_SET_MAP = {
    'v1.0-mini': 'mini_val',
    'v1.0-trainval': 'val',
    'v1.0-test': 'test',
}


def get_nuscenes_scene_splits(version):
    """Return the official scene-name splits for a nuScenes dataset version."""
    from nuscenes.utils import splits

    if version == 'v1.0-trainval':
        return list(splits.train), list(splits.val)
    if version == 'v1.0-test':
        return list(splits.test), []
    if version == 'v1.0-mini':
        return list(splits.mini_train), list(splits.mini_val)
    raise NotImplementedError(f'Unsupported nuScenes version: {version}')


def get_token(info):
    return info.get("token", info.get("metadata", {}).get("token", None))

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names,
                 training=True,
                 root_path=None,
                 logger=None,
                 pkls_path=None,
                 selected_tokens_path=None):
        assert "NUM_POINT_FEATURES" in dataset_cfg
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=pkls_path, logger=logger
        )
        self.infos = []
        self.root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH))
        self.pkls_path = pkls_path if pkls_path is not None else self.root_path
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

        if logger is None:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)


        self.selected_tokens = None

        if selected_tokens_path is not None:
            with open(selected_tokens_path, "r") as f:
                selected_json = json.load(f)

            # supports multi-budget JSON
            # { "10": [...], "100": [...] }
            # OR flat list
            if isinstance(selected_json, dict):
                # default: use biggest budget unless specified
                max_budget = max(selected_json.keys(), key=lambda x: int(x))
                self.selected_tokens = set(selected_json[max_budget])
            else:
                self.selected_tokens = set(selected_json)

            self.logger.info(f"Loaded selected subset: {len(self.selected_tokens)} tokens")

        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)


    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.pkls_path / info_path
            if not info_path.exists():
                self.logger.warning(f'{info_path} does not exist. Make sure you\'ve specified --pkls_path arg correctly')
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        if self.selected_tokens is not None:
            filtered = [
                info for info in nuscenes_infos
                if get_token(info) in self.selected_tokens
            ]
        else:
            filtered = nuscenes_infos

        self.infos.extend(filtered)
        self.logger.info(
            f"Filtered NuScenes dataset: {len(filtered)} / {len(nuscenes_infos)}"
        )

        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos.setdefault(name, []).append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, self.dataset_cfg.NUM_POINT_FEATURES])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, self.dataset_cfg.NUM_POINT_FEATURES])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])

        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict

    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"][:3, :3]
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))

        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size

        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        # SHIFT_COOR origin: https://github.com/PJLab-ADG/3DTrans/
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        if self.dataset_cfg.get('SHIFT_COOR', None):
            input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR


        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    # function origin: https://github.com/PJLab-ADG/3DTrans/
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    # function origin: https://github.com/PJLab-ADG/3DTrans/
    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Truck',
            'bicycle': 'Cyclist',
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'], self.dataset_cfg['FOV_ANGLE']
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    # function origin: https://github.com/PJLab-ADG/3DTrans/
    def nuscene_eval(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except (AssertionError, KeyError):
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=NUSCENES_EVAL_SET_MAP[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    # function origin: https://github.com/PJLab-ADG/3DTrans/
    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.infos)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'nuscenes':
            return self.nuscene_eval(det_annos, class_names, **kwargs)
        else:
            raise NotImplementedError

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10, save_path=None):
        import torch

        gt_database_dir = self.dataset_cfg.get("GT_DATABASE_DIR", f'gt_database_{max_sweeps}sweeps_withvelo')
        database_save_path = save_path / gt_database_dir
        db_info_save_path = save_path / self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH[0]

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                if len(gt_points) == 0:
                    continue

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(save_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))


        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        return all_db_infos


def create_nuscenes_info(version, data_path, max_sweeps=10, logger=None, with_cam=False, selected_tokens_path=None):
    from nuscenes.nuscenes import NuScenes
    from . import nuscenes_utils
    data_path = Path(data_path)
    if selected_tokens_path is not None:
        with open(selected_tokens_path, "r") as f:
            selected_tokens = json.load(f)
    else:
        selected_tokens = None
    train_scenes, val_scenes = get_nuscenes_scene_splits(version)
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    logger.info('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam
    )

    if selected_tokens is not None:
        train_nusc_infos = [x for x in train_nusc_infos if x["token"] in selected_tokens]

    logger.info("train sample: %d, val sample: %d" % (len(train_nusc_infos), len(val_nusc_infos)))

    return train_nusc_infos, val_nusc_infos

def check_if_new_data(dataset_cfg: EasyDict, version: str, data_path: Path, pkls_path: Path, logger) -> bool:
    """Checks if there is new samples in NuScenes dataset that were not added to pkls files.

    :param dataset_cfg: Config of NuScenes dataset in dictionary format
    :param version: version of NuScenes dataset
    :param data_path: path to NuScenes dataset
    :param pkls_path: path to pkls files from dataset
    :raises NotImplementedError: When dataset version is not supported
    :returns: True if all data is not processed.
    """
    from nuscenes.nuscenes import NuScenes

    data_path = Path(data_path)
    pkls_path = Path(pkls_path)

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    train_scenes = nusc.splits['train']
    val_scenes = nusc.splits['val']

    train_val_scenes = list(set(train_scenes + val_scenes))

    train_val_scene_tokens = list(
        map(lambda s: s["token"], filter(lambda s: s["name"] in train_val_scenes, nusc.scene))
    )

    train_val_sample_tokens = list(
        map(lambda s: s["token"], filter(lambda s: s["scene_token"] in train_val_scene_tokens, nusc.sample))
    )

    nuscenes_infos = []
    nuscenes_infos_names_processed = []
    for info_paths in dataset_cfg.INFO_PATH.values():
        for info_path in info_paths:
            info_path = Path(pkls_path) / info_path
            if not info_path.exists() or info_path in nuscenes_infos_names_processed:
                continue
            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)
            nuscenes_infos_names_processed.append(info_path)
    nuscenes_infos_sample_tokens = list(set([s["token"] for s in nuscenes_infos]))

    logger.info(
        "Samples in NuScenes database %d, Samples in pkls files %d." % (len(train_val_sample_tokens), len(nuscenes_infos_sample_tokens))
    )
    return len(train_val_sample_tokens) > len(nuscenes_infos_sample_tokens)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--pkls_path", type=str, help="")
    parser.add_argument("--with_cam", action="store_true", help="include camera infos in pkls")
    parser.add_argument("--subset_json", type=str, default=None)
    args = parser.parse_args()

    logger=common_utils.create_logger()
    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        new_train_infos, new_val_infos = create_nuscenes_info(
            version=args.version,
            data_path=args.data_path,
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            logger=logger,
            with_cam=args.with_cam,
            selected_tokens_path=args.subset_json
        )
        with open(os.path.join(args.pkls_path, dataset_cfg.INFO_PATH["train"][0]), "wb") as f:
            pickle.dump(new_train_infos, f)
        with open(os.path.join(args.pkls_path, dataset_cfg.INFO_PATH["test"][0]), "wb") as f:
            pickle.dump(new_val_infos, f)

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg, None, root_path=Path(args.data_path), logger=logger, training=True, pkls_path=Path(args.pkls_path), selected_tokens_path=args.subset_json
        )
        nuscenes_dataset.infos = new_train_infos
        nuscenes_dataset.create_groundtruth_database(
            max_sweeps=dataset_cfg.MAX_SWEEPS, save_path=Path(args.pkls_path)
        )

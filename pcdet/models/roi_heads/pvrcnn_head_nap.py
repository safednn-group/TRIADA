import torch
import torch.nn as nn

from ...utils import common_utils
from .pvrcnn_head import PVRCNNHead


class NAPPVRCNNHead(PVRCNNHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extract_gts = kwargs.get("model_cfg", {}).get("EXTRACT_GTS", False)

    @staticmethod
    def infer_layers_nap(features, layers):
        activations = []
        for layer in layers:
            features = layer(features)
            if isinstance(layer, nn.ReLU):
                activations.append(features)
        return features, activations

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict,
            nms_config=self.model_cfg.NMS_CONFIG["TRAIN" if self.training else "TEST"],
        )
        if self.training:
            targets_dict = batch_dict.get("roi_targets_dict", None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict["rois"] = targets_dict["rois"]
                batch_dict["roi_labels"] = targets_dict["roi_labels"]

        if self.extract_gts:
            # Ground-truth activation extraction is evaluation-only. Source
            # pretraining uses the standard PVRCNN model.
            assert not self.training
            if (
                batch_dict["gt_boxes"] is not None
                and batch_dict["gt_boxes"].numel() > 0
            ):
                batch_dict["rois"] = batch_dict["gt_boxes"]
            else:
                batch_dict["batch_cls_preds"] = torch.zeros(
                    (batch_dict["batch_size"], 0, 1), device=batch_dict["rois"].device
                )
                batch_dict["batch_box_preds"] = torch.zeros(
                    (batch_dict["batch_size"], 0, self.box_coder.code_size),
                    device=batch_dict["rois"].device,
                )
                batch_dict["activations"] = torch.zeros(
                    (batch_dict["batch_size"], 0, 1024),
                    device=batch_dict["rois"].device,
                )
                batch_dict["cls_preds_normalized"] = False
                return batch_dict

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = (
            pooled_features.permute(0, 2, 1)
            .contiguous()
            .view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)
        )  # (BxN, C, 6, 6, 6)

        shared_features, shared_activations = self.infer_layers_nap(
            pooled_features.view(batch_size_rcnn, -1, 1), self.shared_fc_layer
        )
        rcnn_reg = (
            self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        )  # (B, C)
        rcnn_cls, cls_activations = self.infer_layers_nap(
            shared_features, self.cls_layers
        )
        rcnn_cls = rcnn_cls.transpose(1, 2).contiguous().squeeze(dim=1)
        activations = torch.cat(shared_activations + cls_activations, dim=1).squeeze(
            dim=-1
        )
        if self.extract_gts:
            rcnn_cls = torch.ones_like(rcnn_cls)
        if not self.training:
            batch_cls_preds, batch_box_preds, batch_activations = (
                self.generate_predicted_boxes(
                    batch_size=batch_dict["batch_size"],
                    rois=batch_dict["rois"],
                    cls_preds=rcnn_cls,
                    box_preds=rcnn_reg,
                    activations=activations,
                )
            )
            batch_dict["batch_cls_preds"] = batch_cls_preds
            batch_dict["batch_box_preds"] = batch_box_preds
            batch_dict["activations"] = batch_activations
            batch_dict["cls_preds_normalized"] = False
        else:
            targets_dict["rcnn_cls"] = rcnn_cls
            targets_dict["rcnn_reg"] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict

    def generate_predicted_boxes(
        self, batch_size, rois, cls_preds, box_preds, activations
    ):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)
        batch_activations = activations.view(batch_size, -1, activations.shape[-1])

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(
            -1, code_size
        )

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds, batch_activations

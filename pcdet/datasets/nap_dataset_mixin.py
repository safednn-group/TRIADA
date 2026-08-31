import numpy as np


class NAPDatasetMixin:
    """Attach RoI activations to the standard prediction dictionaries."""

    def generate_prediction_dicts(
        self, batch_dict, pred_dicts, class_names, output_path=None
    ):
        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            # SHIFT_COOR is applied in place below; copy so CPU inference does
            # not mutate the detector's prediction tensor through NumPy.
            pred_boxes = box_dict["pred_boxes"].cpu().numpy().copy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()

            if pred_scores.shape[0] == 0:
                return {
                    "name": np.zeros(0),
                    "score": pred_scores,
                    "boxes_lidar": pred_boxes,
                    "pred_labels": pred_labels,
                    "activations": box_dict["activations"],
                }

            shift_coor = self.dataset_cfg.get("SHIFT_COOR")
            if shift_coor is not None:
                pred_boxes[:, 0:3] -= shift_coor

            return {
                "name": np.asarray(class_names)[pred_labels - 1],
                "score": pred_scores,
                "boxes_lidar": pred_boxes,
                "pred_labels": pred_labels,
                "activations": box_dict["activations"],
            }

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            prediction = generate_single_sample_dict(box_dict)
            prediction["frame_id"] = batch_dict["frame_id"][index]
            prediction["metadata"] = batch_dict["metadata"][index]
            annos.append(prediction)

        return annos

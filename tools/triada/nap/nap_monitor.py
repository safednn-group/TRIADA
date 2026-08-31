import numpy as np
import torch


def split_layerwise(patterns, shapes):
    if patterns.ndim != 2:
        raise ValueError(f"Expected a 2D activation tensor, got {patterns.shape}")
    if sum(shapes) != patterns.shape[1]:
        raise ValueError(
            f"Layer widths sum to {sum(shapes)}, but activations have "
            f"{patterns.shape[1]} features"
        )

    layers = []
    offset = 0
    for shape in shapes:
        layers.append(patterns[:, offset : offset + shape])
        offset += shape
    return layers


def binarize_patterns(layers, quantile=0.0, binarize=True):
    prepared_layers = []
    for layer in layers:
        threshold = torch.quantile(torch.abs(layer), quantile, dim=1, keepdim=True)
        active = torch.abs(layer) > threshold
        if binarize:
            prepared_layers.append(active.to(dtype=layer.dtype))
        else:
            prepared_layers.append(torch.where(active, layer, torch.zeros_like(layer)))

    return torch.cat(prepared_layers, dim=1)


def prepare_patterns(patterns, shapes, quantile=0.0, binarize=True):
    layers = split_layerwise(patterns, shapes)
    return binarize_patterns(layers, quantile=quantile, binarize=binarize)


class DistanceMonitor:
    """Store reference activation patterns and compute configurable p-norm distances."""

    def __init__(
        self,
        class_count,
        device,
        layers_shapes,
        quantile=0.0,
        p_norm=1,
        mean=False,
        **_,
    ):
        self.class_count = class_count
        self.device = device
        self.layers_shapes = (
            layers_shapes if isinstance(layers_shapes, list) else [layers_shapes]
        )
        self.quantile = quantile
        self.p_norm = p_norm
        self.binarize = p_norm == 0
        self.mean = mean
        self.class_patterns_count = {}
        self.known_patterns_set = {}
        self.known_patterns_tensor = {}

    def set_class_patterns_count(self, counts):
        self.class_patterns_count = counts
        for class_id, count in counts.items():
            self.known_patterns_set[class_id] = {
                layer_id: set() for layer_id in range(len(self.layers_shapes))
            }
            self.known_patterns_tensor[class_id] = {
                layer_id: torch.zeros((count, width))
                for layer_id, width in enumerate(self.layers_shapes)
            }

    def add_neuron_pattern(self, neuron_values, labels):
        patterns = prepare_patterns(
            neuron_values,
            self.layers_shapes,
            quantile=self.quantile,
            binarize=self.binarize,
        )
        patterns_np = patterns.cpu().numpy()

        for example_id in range(neuron_values.shape[0]):
            class_id = labels[example_id]
            offset = 0
            for layer_id, width in enumerate(self.layers_shapes):
                pattern_np = patterns_np[example_id, offset : offset + width]
                pattern_key = pattern_np.tobytes()
                known_patterns = self.known_patterns_set[class_id][layer_id]

                if pattern_key not in known_patterns:
                    insert_at = len(known_patterns)
                    self.known_patterns_tensor[class_id][layer_id][insert_at] = (
                        patterns[example_id, offset : offset + width]
                    )
                    known_patterns.add(pattern_key)
                offset += width

    def cut_duplicates(self):
        for class_id, layer_tensors in self.known_patterns_tensor.items():
            for layer_id, patterns in layer_tensors.items():
                unique_count = len(self.known_patterns_set[class_id][layer_id])
                self.known_patterns_tensor[class_id][layer_id] = patterns[:unique_count]

    def to(self, device):
        self.device = device
        for layer_tensors in self.known_patterns_tensor.values():
            for layer_id, patterns in layer_tensors.items():
                layer_tensors[layer_id] = patterns.to(device)
        return self

    def compute_distance(self, neuron_values, class_ids, **_):
        patterns = prepare_patterns(
            neuron_values,
            self.layers_shapes,
            quantile=self.quantile,
            binarize=self.binarize,
        )
        distances = []

        for example_id in range(patterns.shape[0]):
            class_id = class_ids[example_id]
            layer_distances = []
            offset = 0
            for layer_id, width in enumerate(self.layers_shapes):
                known_patterns = self.known_patterns_tensor[class_id][layer_id]
                if known_patterns.numel():
                    distance = torch.cdist(
                        known_patterns,
                        patterns[example_id, offset : offset + width].unsqueeze(0),
                        p=self.p_norm,
                    )
                    level = distance.mean() if self.mean else distance.min()
                else:
                    level = np.iinfo(np.int64).max

                layer_distances.append(int(level))
                offset += width
            distances.append(layer_distances)

        return np.asarray(distances)

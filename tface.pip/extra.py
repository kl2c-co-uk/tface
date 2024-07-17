import tensorflow as tf
import json

# Define the maximum number of bounding boxes you expect
MAX_BBOXES = 10  # Adjust this based on your dataset

# Function to load and preprocess image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image.set_shape([None, None, 3])  # Set dynamic shape for the image
    return image

# Function to parse the JSON file with bounding boxes
def parse_bounding_boxes(json_string):
    data = json.loads(json_string.numpy())
    bboxes = tf.convert_to_tensor(data, dtype=tf.float32)
    padded_bboxes = tf.pad(bboxes, [[0, MAX_BBOXES - tf.shape(bboxes)[0]], [0, 0]], constant_values=0.0)
    padded_bboxes = tf.ensure_shape(padded_bboxes, [MAX_BBOXES, 4])
    return padded_bboxes

# Function to load and preprocess both image and JSON
def load_and_preprocess(image_path, json_path):
    image = load_image(image_path)
    json_string = tf.io.read_file(json_path)
    bounding_boxes = tf.py_function(parse_bounding_boxes, [json_string], tf.float32)
    bounding_boxes.set_shape([MAX_BBOXES, 4])
    return image, bounding_boxes

# List of pairs of image and JSON paths
data_pairs = [
    ("/path/to/image1.jpg", "/path/to/bboxes1.json"),
    ("/path/to/image2.jpg", "/path/to/bboxes2.json"),
    # Add more pairs as needed
]

# Convert the list of pairs into a TensorFlow Dataset
image_paths, json_paths = zip(*data_pairs)
paths_ds = tf.data.Dataset.from_tensor_slices((list(image_paths), list(json_paths)))

# Map the load_and_preprocess function to each element of the dataset
dataset = paths_ds.map(lambda image_path, json_path: load_and_preprocess(image_path, json_path))

# Optional: Batch the dataset
batch_size = 32
dataset = dataset.batch(batch_size)

# Example of iterating over the dataset
for images, bboxes in dataset.take(1):
    print(images.shape, bboxes.shape)












import numpy as np

def generate_anchor_boxes(feature_map_shape, scales, aspect_ratios):
    """
    Generate anchor boxes for each spatial location in the feature map.
    Args:
    - feature_map_shape: Shape of the feature map (height, width).
    - scales: List of scales for the anchor boxes.
    - aspect_ratios: List of aspect ratios for the anchor boxes.

    Returns:
    - anchor_boxes: A numpy array of shape (num_anchors, 4).
    """
    anchors = []
    for scale in scales:
        for aspect_ratio in aspect_ratios:
            w = scale * np.sqrt(aspect_ratio)
            h = scale / np.sqrt(aspect_ratio)
            for i in range(feature_map_shape[0]):
                for j in range(feature_map_shape[1]):
                    cx, cy = j, i
                    anchors.append([cx - w / 2, cy - h / 2, w, h])
    return np.array(anchors)

# Example usage
feature_map_shape = (38, 50)  # Example feature map shape
scales = [128, 256, 512]
aspect_ratios = [0.5, 1, 2]
anchor_boxes = generate_anchor_boxes(feature_map_shape, scales, aspect_ratios)
print(anchor_boxes.shape)

















def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Args:
    - box1: A numpy array of shape (4,) representing [x1, y1, x2, y2].
    - box2: A numpy array of shape (4,) representing [x1, y1, x2, y2].

    Returns:
    - iou: A float representing the IoU between the two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union

def assign_ground_truth_to_anchors(anchor_boxes, ground_truth_boxes, iou_threshold=0.5):
    """
    Assign ground truth bounding boxes to anchor boxes based on IoU.
    Args:
    - anchor_boxes: A numpy array of shape (num_anchors, 4).
    - ground_truth_boxes: A numpy array of shape (num_gt_boxes, 4).
    - iou_threshold: IoU threshold to assign a positive match.

    Returns:
    - labels: A numpy array of shape (num_anchors,) with 1 for positive anchors and 0 for negative anchors.
    - bbox_targets: A numpy array of shape (num_anchors, 4) with the target bounding box deltas.
    """
    num_anchors = anchor_boxes.shape[0]
    labels = np.zeros((num_anchors,), dtype=np.float32)
    bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)

    for i, anchor in enumerate(anchor_boxes):
        best_iou = 0
        best_gt_box = None
        for gt_box in ground_truth_boxes:
            iou = compute_iou(anchor, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_box = gt_box
        if best_iou > iou_threshold:
            labels[i] = 1
            bbox_targets[i] = best_gt_box

    return labels, bbox_targets

# Example usage
ground_truth_boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])  # Example ground truth boxes
labels, bbox_targets = assign_ground_truth_to_anchors(anchor_boxes, ground_truth_boxes)
print(labels.shape, bbox_targets.shape)













def rpn_class_loss(rpn_class_logits, rpn_labels):
    """
    Compute the RPN classification loss.
    Args:
    - rpn_class_logits: Predicted objectness scores (logits) for each anchor.
    - rpn_labels: Ground truth labels for each anchor (1 for object, 0 for background).

    Returns:
    - loss: RPN classification loss.
    """
    rpn_labels = tf.convert_to_tensor(rpn_labels, dtype=tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=rpn_labels, logits=rpn_class_logits)
    return tf.reduce_mean(loss)

def rpn_bbox_loss(rpn_bbox_preds, rpn_bbox_targets):
    """
    Compute the RPN bounding box regression loss.
    Args:
    - rpn_bbox_preds: Predicted bounding box deltas for each anchor.
    - rpn_bbox_targets: Ground truth bounding box deltas for each anchor.

    Returns:
    - loss: RPN bounding box regression loss.
    """
    rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, dtype=tf.float32)
    loss = tf.keras.losses.Huber()(rpn_bbox_targets, rpn_bbox_preds)
    return tf.reduce_mean(loss)

# Example usage
rpn_class_logits = tf.random.normal((anchor_boxes.shape[0],))
rpn_bbox_preds = tf.random.normal((anchor_boxes.shape[0], 4))
class_loss = rpn_class_loss(rpn_class_logits, labels)
bbox_loss = rpn_bbox_loss(rpn_bbox_preds, bbox_targets)
print(class_loss.numpy(), bbox_loss.numpy())









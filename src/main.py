import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def read_input_image(path: str):
    return cv2.imread(path)


def load_labels_csv(
    labels_csv_path: str, read_csv_labels_sep: str,
    read_csv_labels_index_col: str, labels_version: str
):
    return pd.read_csv(
        labels_csv_path, sep=read_csv_labels_sep,
        index_col=read_csv_labels_index_col
    )[labels_version]


def main():
    INPUT_IMAGE_PATH = "./images/success/input-1.jpg" # change to try another image
    INPUT_SHAPE_TARGET_WIDTH = 1028
    INPUT_SHAPE_TARGET_HEIGHT = 1028
    LABELS_VERSION = "OBJECT (2017 REL.)"
    HUB_LOAD_PATH = "./1"
    LABELS_CSV_PATH = "labels.csv"
    READ_CSV_LABELS_SEP = ";"
    READ_CSV_LABELS_INDEX_COL = "ID"
    OUTPUT_IMAGE_PATH = "./images/output.jpg"
    OUTPUT_IMAGE_TRANSPARENT = True
    SCORE_LIMIT = 0.5 # change to try another precision

    img = read_input_image(INPUT_IMAGE_PATH)

    # Resize to respect the input_shape
    inp = cv2.resize(
        img,
        (INPUT_SHAPE_TARGET_WIDTH, INPUT_SHAPE_TARGET_HEIGHT)
    )

    # Converting to uint8
    rgb_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)

    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    # Loading model directly from TensorFlow Hub
    detector = hub.load(HUB_LOAD_PATH)

    labels = load_labels_csv(
        LABELS_CSV_PATH, READ_CSV_LABELS_SEP, READ_CSV_LABELS_INDEX_COL,
        LABELS_VERSION
    )

    # Creating prediction
    boxes, scores, classes, num_detections = detector(rgb_tensor)

    # Processing outputs
    pred_labels = classes.numpy().astype("int")[0]
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype("int")
    pred_scores = scores.numpy()[0]

    img_boxes = None

    # Putting the boxes and labels on the image
    for score, (ymin, xmin, ymax, xmax), label \
            in zip(pred_scores, pred_boxes, pred_labels):
        if score < SCORE_LIMIT:
            continue

        score_txt = f"{100 * round(score)}%"
        img_boxes = cv2.rectangle(
            img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label, (xmin, ymax-10),
                    font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_boxes, score_txt, (xmax, ymax-10),
                    font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

    if img_boxes is None:
        print('No objects founded')
    else:
        plt.imshow(img_boxes)
        plt.savefig(OUTPUT_IMAGE_PATH, transparent=OUTPUT_IMAGE_TRANSPARENT)


main()

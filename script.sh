#!/bin/sh

# Building the dataset
chmod +x slim/datasets/download_mscoco.sh
bash slim/datasets/download_mscoco.sh coco

python slim/datasets/build_visualwakewords_data.py \
    --train_image_dir=coco/raw-data/train2014 \
    --val_image_dir=coco/raw-data/val2014
    --train_annotations_file=coco/raw-data/annotations/instances_train2014.json \
    --val_annotations_file=coco/raw-data/annotations/instances_val2014.json \
    --output_dir=coco/processed \
    --small_object_area_threshold=0.005 \
    --foreground_class_of_interest='person'

# Training the model
python slim/train_image_classifier.py \
    --train_dir=vww_96_grayscale \
    --dataset_name=visualwakewords \
    --dataset_split_name=train \
    --dataset_dir=coco/processed \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --train_image_size=96 \
    --input_grayscale=True \
    --save_summaries_secs=300 \
    --learning_rate=0.045 \
    --label_smoothing=0.1 \
    --learning_rate_decay_factor=0.98 \
    --num_epochs_per_decay=2.5 \
    --moving_average_decay=0.9999 \
    --batch_size=96 \
    --max_number_of_steps=1000000 \
    --clone_on_cpu=True # Note: if you have gpu(cuda available), this argument can be ignore

# Evaluating the model
python slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=vww_96_grayscale/model.ckpt-698580 \
    --dataset_dir=coco/processed/ \
    --dataset_name=visualwakewords \
    --dataset_split_name=val \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --input_grayscale=True \
    --train_image_size=96

# Exporting the model to TensorFlow Lite
## Exporting to a GraphDef protobuf file
python slim/export_inference_graph.py \
    --alsologtostderr \
    --dataset_name=visualwakewords \
    --model_name=mobilenet_v1_025 \
    --image_size=96 \
    --input_grayscale=True \
    --output_file=vww_96_grayscale_graph.pb

## Freezing the weights
python slim/freeze_graph.py \
    --input_graph=vww_96_grayscale_graph.pb \
    --input_checkpoint=vww_96_grayscale/model.ckpt-1000000 \
    --input_binary=true --output_graph=vww_96_grayscale_frozen.pb \
    --output_node_names=MobilenetV1/Predictions/Reshape_1

## Quantizing and converting to TensorFlow Lite
python quantizing_and_converting_to_tflie.py





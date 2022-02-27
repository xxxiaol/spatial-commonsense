# Spatial Commonsense
Source code and data for *Things not Written in Text: Exploring Spatial Commonsense from Visual Signals* (ACL2022 main conference paper).

---

## Dependencies
 - Python>=3.7
 
For pre-trained language model probing:
 - Transformers
 - Pytorch
 - Sklearn
 
For image synthesis:
 - Torchvision
 - Kornia
 - CLIP
 - Taming-transformers
 
For object detection and vision-language model:
 - Scene_graph_benchmark
 - Oscar

## Data
Our datasets are in the `data/` folder.

Size/Height: The objects, text prompts, questions, and lables are in `data.json`. There is an additional pickle file containing the objects in levels.

PosRel: The objects, text prompts and labels for probing are in `data.json`. The questions and answers are in `data_qa.json`.

## Code
The code is in the `code/` folder.
### Image Synthesis
The image synthesis code is adapted from code of Ryan Murdoch, @advadnoun on Twitter.
```
python image_synthesis.py
```
Variables `clip_path` and `taming_path` need to be modified before execution.

Images are generated in `data/{size, height, posrel}/images`. (`{size, height, posrel}` means one of the three words based on the current subtask.)

### Object Detection
Scene_graph_benchmark (VinVL) does not provide code for object detection from custom images directly. 

We first modify `scene_graph_benchmark/tools/mini_tsv/tsv_demo.py` to generate tsv files for our image directory, and run

```
python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR "tools/mini_tsv/{size, height, posrel}" TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True
```
The object detection results are outputed in `predictions.tsv`, and features of bounding boxes are in `feature.tsv`.

### Probing Spatial Commonsense
1. (For Size/Height) Make the depth prediction for each image:
```
python depth_prediction.py
```

2. Image synthesis model probing with bounding boxes in the images:
```
python image_probing_box.py
```

### Solving Natural Language Questions
Reasoning based on the generated images:
1. Generate files required by Oscar+.
```
python build_oscar_data.py
```
Create directories `{size, height, posrel}` under `Oscar/vinvl/datasets`, and then place `oscar_data.json` and `feats.pt` under it.

2. Place `run_vqa.py` in `Oscar/oscar`, and run:
```
python oscar/run_vqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type mask --img_feature_type faster_r-cnn --data_dir vinvl/datasets/{size, height, posrel}/  --model_type bert --model_name_or_path best/best  --task_name vqa_text --do_train --do_lower_case --max_seq_length 128 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 32 --learning_rate 5e-05 --num_train_epochs 25 --output_dir results --label_file vinvl/datasets/vqa/vqa/trainval_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type bce --img_feat_format pt --classifier linear --cls_hidden_scale 3 --txt_data_dir vinvl/datasets/{size, height, posrel}
```

## Citation
Please cite our paper if this repository inspires your work.
```
to be added
```

## Contact
If you have any questions regarding the code, please create an issue or contact the owner of this repository.
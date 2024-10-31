# Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective

This repository contains the implementation and resources for our paper: Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective

The code in this case is modified from OpenAI's guided diffusion repo (https://github.com/openai/guided-diffusion), where you can also download the pretrained diffusion weights (256x256_diffusion.pt).

`image_train.py` is the finetune script for the original image.

`sample_tweedie.py` is the script to sample from the adversarial distribution.

To run the code, please refer to the following bash script: (please modify the path of IMAGE_BASE_DIR and note the format of the folder structure)

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 250 --start_step 150"
IMAGE_BASE_DIR="path/to/your/image/dir"
TMP_PATH="./tmp"

rm -rf $TMP_PATH
for i in {0..999} ; do
  CUR_DIR="$IMAGE_BASE_DIR/$i"
  min_filename=$(ls $CUR_DIR | sort | head -1)
  min_filepath="$CUR_DIR/$min_filename"
  python image_train.py --data_dir ./data --model_path ./models/256x256_diffusion.pt $MODEL_FLAGS --save_interval 300 --lr 1e-6 --image_label $i --image_path $min_filepath --tmp_path $TMP_PATH
  for j in {0..5} ; do
    python sample_tweedie.py $MODEL_FLAGS --classifier_scale 5.0 --classifier_path models/256x256_classifier.pt --model_path $TMP_PATH/model000300.pt $SAMPLE_FLAGS --image_path $min_filepath --image_label $i --save_path ./save --sample_id $j
  done
  rm -rf $TMP_PATH
done
```


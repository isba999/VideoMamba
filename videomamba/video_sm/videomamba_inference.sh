# path to video for visualization
VIDEO_PATH='/media/aimac/DATA/data/Datasets/Collected/voxceleb_preprocessing/VoxCeleb2_train/id00015/0fijmz4vTVU/chunk_videos/0fijmz4vTVU#00001#278-456.webm'
# path to pretrain model
MODEL_PATH='/media/aimac/DATA/data/Projects/VideoMamba/videomamba/video_sm/checkpoints/videomamba_m16_k400_mask_pt_f8_res224.pth'

python3 videomamba_inference.py \
    --input_path ${VIDEO_PATH} \
    --model_path ${MODEL_PATH} \
    --mask_ratio 0.75 \
    --mask_type tube \
    --model_name videomamba_middle_pretrain

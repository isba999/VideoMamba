# Based on VideoMAE visualization script : https://github.com/MCG-NJU/VideoMAE/blob/main/run_videomae_vis.py

import argparse
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from models import videomamba_pretrain
import utils
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from datasets.build import build_pretraining_dataset
from torchvision.transforms import ToPILImage
from einops import rearrange
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets.masking_generator import TubeMaskingGenerator # wont be used i guess but just for testing
from datasets.transforms import *
from pathlib import Path

class DataAugmentationForVideoMamba(object):
    '''
    This class implements data augmentation for VideoMamba, following a similar pipeline as VideoMAE. It includes : 
    - Normalization using ImageNet's default mean and standard deviation.
    - Central cropping for input resizing.
    - Conversion of images to tensors.
    '''
    
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        #self.masked_position_generator = None

    def __call__(self, images):
        process_data , _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def get_args():
    parser = argparse.ArgumentParser('VideoMamba visualization reconstruction script', add_help=False)
    parser.add_argument('--input_path',
                        type=str, help='input video path')
    parser.add_argument('--save_path', default = 'reconstruction results',
                        type=str, help='save video path')
    parser.add_argument('--model_path',
                        type=str, help='checkpoint path')
    parser.add_argument('--mask_type', default='tube', choices=['attention', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--mask_ratio', default=0.75, 
                        type=float, help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--num_frames', default=8,
                        type=int)
    parser.add_argument('--decoder_embed_dim', default=576, 
                        type=int, help='embedding dimension of decoder')
    parser.add_argument('--input_size', default=224, 
                        type=int, help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        type=str,help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--model_name', default='videomamba_middle_pretrain', type=str, metavar='MODEL',
                        help='Name of model to vis')
    
    return parser.parse_args()
    

def get_model(args): 
    """
    Create the model using the specified architecture & parameters
    """
    print(f'Creating model : {args.model_name}')
    model = create_model(
        model_name=args.model_name,
        pretrained=False,                 # No pretrained weights
        #drop_path_rate=0.4,               # Matches pretraining script
        clip_decoder_embed_dim=args.decoder_embed_dim,       # Matches pretraining script
        num_frames=args.num_frames,                     # Matches pretraining script
    )
    return model

def main(args):

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Set the device
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    # Initialize the model
    model = get_model(args)

    # Define the window size
    patch_size = model.patch_embed.patch_size  # Spatial patch size (e.g., 16x16 for 224x224 input)
    window_size = (args.num_frames, args.input_size // patch_size[0], args.input_size // patch_size[1]) # Temporal units (num_frames grouped by tubelet_size)
    print(window_size) # -> (8, 14, 14) -> 8 temporal units (because the tubelet size=1) each contains 14*14 patches
    args.patch_size = patch_size # Add window_size and patch_size to args so they can be accessed globally without needing to pass them separately.
    args.window_size = window_size
    
    # Move model to device
    model.to(device)

    # Load the checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')

    # Load the model's weights
    model.load_state_dict(checkpoint)
    
    # Set the model to the evaluation mode
    model.eval()

    # Compare model and checkpoint state dictionaries
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint

    missing_keys = [k for k in model_state_dict if k not in checkpoint_state_dict]
    unexpected_keys = [k for k in checkpoint_state_dict if k not in model_state_dict]

    if not missing_keys and not unexpected_keys:
        print("All keys match between model and checkpoint.")
    else:
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    # Load the input video
    with open(args.input_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    
    # Process the frames
    frame_id_list = np.linspace(0, len(vr) - 1, args.num_frames, dtype=int).tolist()
    video_data = vr.get_batch(frame_id_list).asnumpy()
    #print(video_data.shape) # -> (8, 360, 640, 3) -> 8 frames, each having a height of 360 and a width of 640, with 3 color channels (RGB)
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)] # Converts each frame in video_data (selected via frame_id_list) to a PIL image in RGB format and stores them in a list
    transforms = DataAugmentationForVideoMamba(args)
    img, bool_masked_pos = transforms((img, None))
    #print(type(img)) # -> Torch Tensor
    print(bool_masked_pos.shape) # -> 1568, which represents the total number of patches across 8 frames : 8*14*14.
    #print(img.shape) # -> # (24, 224, 224): 8 frames x 3 color channels, resized to 224x224.
    #print(img.size()[-2:]) # Get the height and width of the image (last two dimensions of the tensor)
    img = img.view((args.num_frames, 3) + img.size()[-2:]) # -> (8, 3, 224, 224)
    img = img.transpose(0, 1) # -> (3, 8, 224, 224)
    bool_masked_pos = torch.from_numpy(bool_masked_pos) # -> Torch Tensor

    with torch.no_grad():
        # Add batch dimension to the image tensor -> (1, 3, 8, 224, 224)
        img = img.unsqueeze(0) 
        print(f"Image shape after unsqueeze: {img.shape}")

        # Add batch dimension to the mask tensor -> (1, 784)
        bool_masked_pos = bool_masked_pos.unsqueeze(0) 
        print(f"Mask shape after unsqueeze: {bool_masked_pos.shape}")

        # Move frames to the specified device
        img = img.to(device, non_blocking=True)

        # Move mask to the device before any operations
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        # Ensure mask is Boolean
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)  # Flatten and convert to Boolean

        # Add one extra slot for the CLS token (value: False) and ensure it's on the same device
        cls_token_mask = torch.zeros((bool_masked_pos.size(0), 1), device=device, dtype=torch.bool)
        bool_masked_pos = torch.cat([cls_token_mask, bool_masked_pos], dim=1)
        print(f"Mask shape after CLS token adjustment: {bool_masked_pos.shape}")  # -> (1, 1569)

        # Inference
        outputs = model(img, bool_masked_pos)
        print(outputs.shape) # -> (1, 1, 393, 512)
        print('Inference completed successfully.')

if __name__ == '__main__':
    opts = get_args()
    main(opts)
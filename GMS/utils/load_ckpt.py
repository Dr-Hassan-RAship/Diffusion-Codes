#[CHANGED] --> Instead of storing the ckpt file in SD-VAE-weights folder, we directly download the ckpt file and store the
# ckpt file in the hugging face cache directory. We can then access the ckpt file using the torch.load function.
# or we can get the state_dict from the ckpt file and then load the state_dict into the vae models

# pip install -U "huggingface_hub[hf_xet]"

from huggingface_hub import hf_hub_download # [CHANGED] --> importing huggingface_hub for direct download of the VAE ckpt file
import torch
from diffusers  import AutoencoderTiny

def get_state_dict(ckpt_url = 'https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/768-v-ema.ckpt', 
                   repo_id  = 'stabilityai/stable-diffusion-2', 
                   filename = '768-v-ema.ckpt'):
    
    # Check out this 
    
    # Download the checkpoint file
    ckpt_path_local = hf_hub_download(repo_id=repo_id, filename=filename)

    print(f"Checkpoint downloaded to: {ckpt_path_local}")

    # Load the state dictionary from the checkpoint file
    # This loads the dictionary of weights, but doesn't load the models themselves
    try:
        state_dict = torch.load(ckpt_path_local, map_location="cpu")
        print("Successfully loaded state dictionary.")
        # You can inspect the keys if you want to see what's inside
        # print(state_dict.keys()) # Uncomment to see the keys
    except Exception as e:
        print(f"Error loading state dictionary: {e}")

def get_tiny_autoencoder(device = 'cuda'):
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).to(device).eval()
    
    # freeze all params
    for param in vae.parameters():
        param.requires_grad = False
    
    return vae
    
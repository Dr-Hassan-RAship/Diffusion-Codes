# ------------------------------------------------------------------------------#
# File name      : valid.py
# Purpose        : Validation/inference for Latent Diffusion-based Medical Segmentation
# Usage          : python valid.py --config ./configs/your_valid.yaml
# Authors        : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Last Modified  : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Package imports ----------------------------------#
import os, time, logging, argparse
import numpy as np
import pandas as pd
import torch
from tqdm               import tqdm
from PIL                import Image
from einops             import rearrange
from torch.utils.data   import DataLoader
from sklearn.metrics    import confusion_matrix

# --------------------------- Project modules ----------------------------------#
from data.image_dataset                 import Image_Dataset
from utils.tools                        import seed_reproducer, load_checkpoint, get_cuda, print_options
from utils.get_logger                   import open_log
from utils.load_pretrained_models       import load_pretrained_model
from networks.latent_mapping_model      import ResAttnUNet_DS
from networks.models.autoencoder        import AutoencoderKL
from networks.models.distributions      import DiagonalGaussianDistribution

# --------------------------- Utility functions --------------------------------#

def load_img(path, img_size=224):
    """Loads and normalizes a grayscale mask image to [0,1], resizes to (img_size, img_size)."""
    image = Image.open(path).convert("L").resize((img_size, img_size), resample=Image.NEAREST)
    image = np.array(image).astype(np.float32) / 255.0
    return image

def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    """Extracts mean and logvar from encoder posterior."""
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not implemented")
    return scale_factor * mean, logvar

def vae_decode(vae_model, pred_mean, scale_factor):
    """Decodes predicted latent, clamps to [0,1], returns grayscale segmentation."""
    z = 1.0 / scale_factor * pred_mean
    pred_seg = vae_model.decode(z).sample
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True)
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0)
    return pred_seg

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/kvasir-seg_valid.yaml', type=str,
                        help='Path to the validation config YAML')
    return parser.parse_args()

# --------------------------- Main validation loop -----------------------------#
def run_validator():
    args    = arg_parse()
    configs = yaml.safe_load(open(args.config))
    # ---- Resolve paths (prefer your load_config if available) ----
    snapshot_path = configs['paths']['snapshot']
    log_path      = os.path.join(snapshot_path, 'logs')
    save_seg_img_path = configs['paths']['predicted_masks']

    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs(save_seg_img_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    gpus = ','.join([str(i) for i in configs['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    seed_reproducer(configs['seed'])
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # ---- DataLoader ----
    valid_dataset    = Image_Dataset(configs['paths']['pickle_file'], stage='test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    # ---- Load Models ----
    mapping_model = get_cuda(ResAttnUNet_DS(
        in_channel     = configs['model']['in_channel'],
        out_channels   = configs['model']['out_channels'],
        num_res_blocks = configs['model']['num_res_blocks'],
        ch             = configs['model']['ch'],
        ch_mult        = configs['model']['ch_mult']
    ))
    mapping_model = load_checkpoint(mapping_model, configs['paths']['model_weight'])
    mapping_model.eval()

    vae_model = load_pretrained_model(
        model_cls               = AutoencoderKL,   # or AutoencoderTiny or any compatible model
        pretrained_name_or_path = configs.get('vae_pretrained', "madebyollin/taesd"),
        dtype                   = getattr(torch, configs.get('precision', 'float32')),
        device                  = "cuda",
        freeze                  = True,
    )
    scale_factor = float(configs.get('vae_scale_factor', 1.0))

    # ---- Validation loop ----
    mse_loss    = torch.nn.MSELoss(reduction='mean')
    epoch_start = time.time()
    name_list, T_loss_valid = [], []

    for batch_data in tqdm(valid_dataloader, desc='Valid: '):
        img_rgb = batch_data['img'] / 255.0
        img_rgb = 2. * img_rgb - 1.
        seg_raw = batch_data['seg'].permute(0, 3, 1, 2) / 255.0
        seg_rgb = 2. * seg_raw - 1.
        seg_img = torch.mean(seg_raw, dim=1, keepdim=True)
        name    = batch_data['name'][0]
        name_list.append(name)

        with torch.no_grad():
            img_latent_mean = vae_model.encode(get_cuda(img_rgb)).latents
            seg_latent_mean = vae_model.encode(get_cuda(seg_rgb)).latents
            out_latent_mean_dict = mapping_model(img_latent_mean)

            loss_Rec = configs['loss']['w_rec'] * mse_loss(out_latent_mean_dict['out'], seg_latent_mean)
            pred_seg = vae_decode(vae_model, out_latent_mean_dict['out'], scale_factor)
            pred_seg = pred_seg.repeat(1, 3, 1, 1)

            x_sample = rearrange(pred_seg.squeeze().cpu().numpy(), 'c h w -> h w c')
            x_sample = np.where(x_sample > 0.5, 1, 0) * 255.
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(save_seg_img_path, name + '.jpg'))

            T_loss_valid.append(loss_Rec.item())

    T_loss_valid = np.mean(T_loss_valid)
    logging.info("Valid:\nloss: {:.4f}".format(T_loss_valid))

    # ---- Metrics calculation ----
    csv_path  = os.path.join(snapshot_path, 'results.csv')
    pred_path = save_seg_img_path
    true_path = os.path.join(os.path.dirname(configs['paths']['pickle_file']), 'masks')

    name_list = sorted(os.listdir(pred_path))
    name_list = [x.replace('.jpg', '').replace('_segmentation', '') for x in name_list]

    dsc_list, iou_list = [], []

    for case_name in tqdm(name_list):
        seg_pred = load_img(os.path.join(pred_path, case_name + '.jpg'))
        seg_true = load_img(os.path.join(true_path, case_name + '.jpg'))

        preds = seg_pred.reshape(-1)
        gts   = seg_true.reshape(-1)

        y_pre  = np.where(preds >= 0.5, 1, 0)
        y_true = np.where(gts   >= 0.5, 1, 0)

        # Defensive for confusion_matrix shape
        cm = confusion_matrix(y_true, y_pre, labels=[0, 1])
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:  # All predictions/labels are same (rare case)
            TN = FP = FN = TP = 0

        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) else 0
        miou     = float(TP) / float(TP + FP + FN) if (TP + FP + FN) else 0

        dsc_list.append(f1_or_dsc)
        iou_list.append(miou)

    # Add mean/std
    name_list.extend(['Avg', 'Std'])
    dsc_list.extend([np.mean(dsc_list), np.std(dsc_list, ddof=1)])
    iou_list.extend([np.mean(iou_list), np.std(iou_list, ddof=1)])

    df = pd.DataFrame({'Name': name_list, 'DSC': dsc_list, 'IoU': iou_list})
    df.to_csv(csv_path, index=False)

    logging.info("DSC: {:.4f}, IOU: {:.4f}".format(dsc_list[-2], iou_list[-2]))
    logging.info('Time Taken: %d sec' % (time.time() - epoch_start))
    logging.info('\n')

if __name__ == '__main__':
    run_validator()

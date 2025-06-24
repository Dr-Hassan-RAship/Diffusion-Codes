# ------------------------------------------------------------------------------#
#
# File name                 : valod.py
# Purpose                   : Main Inference Loop for GMS
# Usage                     : python valid.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch, os, time, logging, argparse

import numpy            as np
import pandas           as pd

from tqdm               import tqdm
from PIL                import Image
from einops             import rearrange
from torch.utils.data   import DataLoader
from sklearn.metrics    import confusion_matrix

from data               import *
from utils              import *
from network            import *
from diffusers          import AutoencoderTiny
from configs.config     import *

# --------------------------- Main validation loop -----------------------------#
def run_validator():
    # ----- Get config and make relevant paths -----#
    log_path          = os.path.join(SNAPSHOT_PATH, 'logs')
    save_seg_img_path = PRED_MASKS_PATH

    os.makedirs(SNAPSHOT_PATH, exist_ok = True)
    os.makedirs(save_seg_img_path, exist_ok = True)
    os.makedirs(log_path, exist_ok = True)
    
    # ----- Hardware, seed & Precision -----#
    gpus = ','.join([str(i) for i in GPUS])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    seed_reproducer(SEED)
    np_dtype, torch_dtype = get_precision_dtypes(PRECISION)

    # --------- Logging setup -----#
    open_log('train', log_path)
    logging.info()
    print_options()

    # ---- DataLoader ----
    valid_dataset    = Image_Dataset(PICKLE_FILE_PATH, stage = 'test')
    valid_dataloader = DataLoader(valid_dataset, batch_size = 1, pin_memory = True, drop_last = False, shuffle = False)

    # ---- Load Models with required device and dtype ----
    mapping_model = get_cuda(
        ResAttnUNet_DS(**MODEL_PARAMS)
    ).to(dtype = torch_dtype)

    mapping_model = load_checkpoint(mapping_model, MODEL_WEIGHT_PATH)
    mapping_model.eval()

    # Modular VAE loading (dtype, device, freeze are all config-controlled)
    vae_model = load_pretrained_model(
        model_cls               = AutoencoderTiny,
        pretrained_name_or_path = "madebyollin/taesd",
        dtype                   = torch_dtype,
        device                  = "cuda",
        freeze                  = True,
        eval_mode               = True
    )
    vae_model.eval()
    scale_factor = VAE_SCALE_FACTOR

    # ---- Validation loop ----
    mse_loss    = torch.nn.MSELoss(reduction='mean')
    epoch_start = time.time()
    name_list, T_loss_valid = [], []

    for batch_data in tqdm(valid_dataloader, desc = 'Valid: '):
        img_rgb = batch_data['img'] / 255.0
        img_rgb = 2. * img_rgb - 1.
        seg_raw = batch_data['seg'].permute(0, 3, 1, 2) / 255.0
        seg_rgb = 2. * seg_raw - 1.
        name    = batch_data['name']
        name_list.append(name)

        with torch.no_grad():
            img_latent_mean      = vae_model.encode(get_cuda(img_rgb)).latents
            seg_latent_mean      = vae_model.encode(get_cuda(seg_rgb)).latents
            out_latent_mean_dict = mapping_model(img_latent_mean)

            loss_Rec = W_REC * mse_loss(out_latent_mean_dict['out'], seg_latent_mean)
            pred_seg = vae_decode(vae_model, out_latent_mean_dict['out'], scale_factor)
            pred_seg = pred_seg.repeat(1, 3, 1, 1)

            x_sample = rearrange(pred_seg.squeeze().cpu().numpy(), 'c h w -> h w c')
            x_sample = np.where(x_sample > 0.5, 1, 0) * 255.
            img      = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(save_seg_img_path, name + IMG_FORMAT))

            T_loss_valid.append(loss_Rec.item())

    T_loss_valid = np.mean(T_loss_valid)
    logging.info("Valid:\nloss: {:.4f}".format(T_loss_valid))

    # ---- Metrics calculation ----
    csv_path  = os.path.join(SNAPSHOT_PATH, 'results.csv')
    pred_path = save_seg_img_path
    true_path = os.path.join(os.path.dirname(PICKLE_FILE_PATH), 'masks')

    name_list = sorted(os.listdir(pred_path))
    name_list = [x.replace(IMG_FORMAT, '').replace('_segmentation', '') for x in name_list]

    dsc_list, iou_list = [], []

    for case_name in tqdm(name_list):
        seg_pred = load_img(os.path.join(pred_path, case_name + IMG_FORMAT), img_size = IMG_SIZE, dtype_resize = np_dtype)
        seg_true = load_img(os.path.join(true_path, case_name + IMG_FORMAT), img_size = IMG_SIZE, dtype_resize = np_dtype)

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

# -----------------------------------------------------------------------#
if __name__ == "__main__":
    run_validator()

# -------------------------------- End ----------------------------------#

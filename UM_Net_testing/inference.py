#------------------------------------------------------------------------------#
#
# File name         : inference.py
# Purpose           : Runs inference on test subjects for the Qatar Covid-19 dataset
# Usage (command)   : python inference.py --config config_{method}.yaml
#
# Authors           : Syed Muqeem Mahmood, Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 24100025@lums.edu.pk, 21060003@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date         : October 7, 2024
#
#------------------------------------------------------------------------------#

import os, torch, csv, cv2, tqdm, h5py, random
import numpy                        as np
import glob                         as glob

from torchvision                    import transforms
from torch.utils.data               import DataLoader

from helpers.fom                    import test_single_slice
from helpers.parse_yaml             import parse_yaml_config
from UM_Net                         import *
from dataloaders.dataset            import *

from torch.nn                       import functional               as Fun
from torchvision.transforms         import functional               as F

#------------------------------------------------------------------------------#
class inference_routine:
    def __init__(self, config = None, config_file = None):

        print('---------------------------------')
        print('> Initializing Inference Session <')
        print('---------------------------------')

        if config is None:
            assert config_file is not None, f"`config_file` is needed if `config` not provided."
            assert os.path.exists(config_file), f"config file {config_file} not found."
            config = parse_yaml_config(config_file)

        #-------------------------------------------------------------#
        config_inf              = config['inference']
        self.experiment_name    = config_inf.get('experiment_name')
        self.epoch_choice       = config_inf.get('epoch_choice')
        self.slice_size         = config_inf.get('slice_size')
        self.test_root_path     = config_inf.get('input_folder')
        self.output_folder      = config_inf.get('output_folder')
        self.num_classes        = config_inf.get('num_classes')
        self.num_workers        = config_inf.get('num_workers')
        self.input_channels     = config_inf.get('input_channels')
        self.net_model          = config_inf.get('net_model')
        
    #------------------------------------------------------------------------------#
    def predict(self):

        model           = UM_Net(num_classes = self.num_classes).cuda()

        if not self.epoch_choice:
            model_path = os.path.join('snapshot', self.experiment_name, 'models', 'best_model.pth')
            print ('\n -------- Loading Best Model Weights -------- \n')
        else:
            model_path = os.path.join('snapshot', self.experiment_name, 'models', 'epoch_num_{}.pth'.format(self.epoch_choice))
            print ('\n -------- Loading Epoch Weights -------- \n')

        model.load_state_dict(torch.load(model_path, weights_only = True)); model.eval()

        #----------------------------------------------------------------------#
        # Test Data. We use ValGenerator because the test dataset is (also) 2D.
        db_test        = Polyp_Dataset(root         = self.test_root_path,
                                       mode        = "test",
                                       slice_size  = self.slice_size)

        def worker_init_fn(worker_id): random.seed(self.seed + worker_id)

        testloader     = DataLoader(db_test,
                                    batch_size   = 1,
                                    shuffle      = False,
                                    num_workers  = self.num_workers)
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(testloader):
                patient_id, volume_batch, label_batch = sampled_batch['file_name'], sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch   = volume_batch.cuda(), label_batch.cuda()

                outputs                     = model(volume_batch)
                outputs_soft                = torch.sigmoid(outputs)
                pred                        = (outputs_soft > 0.5).squeeze(0).detach().cpu().numpy()
                mask                        = label_batch[:, 0, :, :].squeeze(0).detach().cpu().numpy()
                dice_score, IoU_score       = test_single_slice(pred, mask)
                
                # print(patient_id, dice_score, IoU_score)
                csv_logger.writerow([patient_id, dice_score, IoU_score])
                csvfile.flush()

#------------------------------------------------------------------------------#
if __name__== '__main__':

    import argparse

    parser          = argparse.ArgumentParser(description = 'Inference over Kvasir-SEG dataset')
    parser.add_argument('--config', type = str, required = True, help = '.yaml config file')
    args            = parser.parse_args()

    inference_obj   = inference_routine(config_file = args.config)

    output_folder   = inference_obj.output_folder
    experiment_name = inference_obj.experiment_name

    snapshot_path   = "{}/{}".format(inference_obj.output_folder, experiment_name)

    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

    with open(snapshot_path + '/inference_results.csv', 'a') as csvfile:
        csv_logger = csv.writer(csvfile)
        csv_logger.writerow(['id', 'dice', 'IoU'])
        inference_obj.predict()

# ---------------------------------------------------------------------------- #

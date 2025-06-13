#------------------------------------------------------------------------------#
# File name         : inference.py
# Purpose           : Runs inference on test subjects for the MM2 Cardiac Dataset
# Usage (command)   : python inference.py --config config_{method}.yaml
#
# Authors           : Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 21060003@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date         : April 23, 2025
#------------------------------------------------------------------------------#

import os, sys, math, torch, csv
import numpy                        as np
import nibabel                      as nib
import torch.nn                     as nn

from skimage                        import transform                as sktform
from scipy.ndimage                  import zoom

from helpers.inference_utils        import *
from helpers.parse_yaml             import parse_yaml_config
from helpers.train_utils            import compute_sdf

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
        config_inf                  = config['inference']

        self.experiment_name        = config_inf.get('experiment_name')
        self.epoch_choice           = config_inf.get('epoch_choice')
        self.labeled_ratio          = config_inf.get('labeled_ratio')

        self.input_folder           = config_inf.get('input_folder')
        self.output_folder          = config_inf.get('output_folder')

        self.num_classes            = config_inf.get('num_classes')
        self.input_channels         = config_inf.get('input_channels')
        self.crop_factor            = config_inf.get('crop_factor')
        self.resolution             = config_inf.get('resolution')

        self.save_FOMs_report       = config_inf.get('save_FOMs_report')
        self.save_segmaps           = config_inf.get('save_segmaps')
        self.save_softmax_probs     = config_inf.get('save_softmax_probs')
        self.save_lsf               = config_inf.get('save_lsf')

        self.do_preprocess          = config_inf.get('do_preprocess')
        self.do_postprocess         = config_inf.get('do_postprocess')
        self.get_FOMs               = config_inf.get('get_FOMs')

        #-------------------------------------------------------------#
        config_net                  = config['network']
        self.initialize             = config_net.get('initialization')

    #------------------------------------------------------------------------------#
    def predict(self):
        net             = net_factory(self)

        if not self.epoch_choice:
            model_path  = os.path.join('snapshot', self.labeled_ratio, self.experiment_name, 'best_model.pth')
            print ('\n -------- Loading Best Model Weights -------- \n')
        else:
            model_path  = os.path.join('snapshot', self.labeled_ratio, self.experiment_name, 'epoch_num_{}.pth'.format(self.epoch_choice))
            print ('\n -------- Loading Best Epoch Weights -------- \n')

        net.load_state_dict(torch.load(model_path)); net.eval()

        for case in os.listdir(self.input_folder):
            # Loading SA (ED+ES) images for each subject
            sa_ed_path  = nib.load(os.path.join(self.input_folder, case, case + '_SA_ED.nii.gz'))
            sa_es_path  = nib.load(os.path.join(self.input_folder, case, case + '_SA_ES.nii.gz'))

            sa_ed_affine, sa_ed_header, sa_ed_pix_size, sa_ed_shape, sa_ed = read_nifti_image(sa_ed_path)
            sa_es_affine, sa_es_header, sa_es_pix_size, sa_es_shape, sa_es = read_nifti_image(sa_es_path)

            #------------------------------------------------------------------#
            # Preprocessing SA images
            if self.do_preprocess:
                sa_ed = minmax_norm(sa_ed)
                sa_es = minmax_norm(sa_es)
                sa_ed = zoom(sa_ed, (self.crop_factor[0]/sa_ed_shape[0], self.crop_factor[1]/sa_ed_shape[1], 1.0), order=0)
                sa_es = zoom(sa_es, (self.crop_factor[0]/sa_es_shape[0], self.crop_factor[1]/sa_es_shape[1], 1.0), order=0)

            #------------------------------------------------------------------#
            if 'dtc' in self.experiment_name:
                sa_ed_seg, sa_ed_prob, out_tanh1 = self.predict_nifti(net, sa_ed, sa_ed_shape, sa_es_shape)
                sa_es_seg, sa_es_prob, out_tanh2 = self.predict_nifti(net, sa_es, sa_ed_shape, sa_es_shape)
            else:
                sa_ed_seg, sa_ed_prob = self.predict_nifti(net, sa_ed, sa_ed_shape, sa_es_shape)
                sa_es_seg, sa_es_prob = self.predict_nifti(net, sa_es, sa_ed_shape, sa_es_shape)

            #------------------------------------------------------------------#
            # if self.save_lsf and 'dtc' in self.experiment_name:
            #     directory = os.path.join(self.output_folder, self.labeled_ratio, self.experiment_name, case)
            #     if not os.path.exists(directory): os.makedirs(directory)
            #
            #     sa_ed_tanh_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder,
            #                                                                       self.labeled_ratio,
            #                                                                       self.experiment_name,
            #                                                                       case, 'SA_ED_TANH')
            #
            #     sa_ed_tanh              = nib.Nifti1Image(out_tanh1.cpu().detach().numpy(),
            #                                               sa_ed_affine, header=sa_ed_header)
            #     nib.save(sa_ed_tanh, sa_ed_tanh_path)
            #
            #     sa_es_tanh_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder,
            #                                                                       self.labeled_ratio,
            #                                                                       self.experiment_name,
            #                                                                       case, 'SA_ES_TANH')
            #
            #     sa_es_tanh              = nib.Nifti1Image(out_tanh2.cpu().detach().numpy(),
            #                                               sa_es_affine, header=sa_es_header)
            #     nib.save(sa_es_tanh, sa_es_tanh_path)

            if self.save_segmaps:
                directory = os.path.join(self.output_folder, self.labeled_ratio, self.experiment_name, case)
                if not os.path.exists(directory): os.makedirs(directory)

                sa_ed_pred_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder,
                                                                                  self.labeled_ratio,
                                                                                  self.experiment_name,
                                                                                  case, 'SA_ED_pred')

                sa_ed_pred              = nib.Nifti1Image(sa_ed_seg.astype(np.uint8),
                                                          sa_ed_affine, header=sa_ed_header)
                nib.save(sa_ed_pred, sa_ed_pred_path)

                sa_es_pred_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder,
                                                                                  self.labeled_ratio,
                                                                                  self.experiment_name,
                                                                                  case, 'SA_ES_pred')

                sa_es_pred              = nib.Nifti1Image(sa_es_seg.astype(np.uint8),
                                                          sa_es_affine, header=sa_es_header)
                nib.save(sa_es_pred, sa_es_pred_path)

            if self.save_softmax_probs:
                directory = os.path.join(self.output_folder, self.labeled_ratio, self.experiment_name, case)
                if not os.path.exists(directory): os.makedirs(directory)

                sa_ed_pred_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder,
                                                                                  self.labeled_ratio,
                                                                                  self.experiment_name,
                                                                                  case, 'SA_ED_prob')

                sa_ed_prob              = nib.Nifti1Image(sa_ed_prob, sa_ed_affine, header=sa_ed_header)
                nib.save(sa_ed_prob, sa_ed_pred_path)

                sa_es_pred_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder,
                                                                                  self.labeled_ratio,
                                                                                  self.experiment_name,
                                                                                  case, 'SA_ES_prob')

                sa_es_prob              = nib.Nifti1Image(sa_es_prob, sa_es_affine, header=sa_es_header)
                nib.save(sa_es_prob, sa_es_pred_path)

    #--------------------------------------------------------------------------#
    def predict_nifti(self, net, sa_img, sa_ed_shape, sa_es_shape):

        SA_image    = []
        for j in range(sa_img.shape[2]):
            sa_img_slice = sa_img[:,:,j]
            sa_img_slice = np.expand_dims(sa_img_slice, axis = 0)
            SA_image     += [sa_img_slice]

        SA_image    = np.stack(SA_image,axis=0)
        SA_image    = torch.Tensor(SA_image).cuda()

        experiments = ['FS_100p', 'FS_10p', 'FS_20p', 'ACMT-Ent', 'ACMT-MU',
                       'ACMT-PErr', 'ACMT-SErr', 'uamt', 'mean_teacher',
                       'dae_mt', 'slc']

        if any(substring in self.experiment_name for substring in experiments):
            output      = net(SA_image)
            softOutput  = torch.softmax(output, dim = 1).cpu().detach().numpy()     # (10, 4, 256, 256)

        if 'urpc' in self.experiment_name:
            output, _, _, _  = net(SA_image)
            softOutput  = torch.softmax(output, dim = 1).cpu().detach().numpy()     # (10, 4, 256, 256)

        if 'MCNet' == self.experiment_name:
            output1, output2 = net(SA_image)
            #softOutput = torch.softmax(output1, dim = 1).cpu().detach().numpy()
            avg_output  = (output1 + output2) / 2
            softOutput  = torch.softmax(avg_output, dim = 1).cpu().detach().numpy()

        if 'Aux_Dec' == self.experiment_name:
            output1, output2 = net(SA_image)
            #softOutput = torch.softmax(output1, dim = 1).cpu().detach().numpy()
            avg_output  = (output1 + output2) / 2
            softOutput  = torch.softmax(avg_output, dim = 1).cpu().detach().numpy()

        if 'MCNet_plus' == self.experiment_name:
            output1, output2, output3 = net(SA_image)
            # softOutput  = torch.softmax(output1, dim = 1).cpu().detach().numpy()
            avg_output  = (output1 + output2 + output3) / 3
            softOutput  = torch.softmax(avg_output, dim = 1).cpu().detach().numpy()

        if 'mutual_reliable' == self.experiment_name:
            output, outputfeatures = net(SA_image)
            softOutput  = torch.softmax(output[0], dim = 1).cpu().detach().numpy()
            # avg_output  = (output[0] + output[1]) / 2
            # softOutput  = torch.softmax(avg_output, dim = 1).cpu().detach().numpy()

        if 'dtc' in self.experiment_name:
            out_tanh, output = net(SA_image)
            softOutput  = torch.softmax(output, dim = 1).cpu().detach().numpy()     # (10, 4, 256, 256)

        if 'sassnet' in self.experiment_name:
            out_tanh, output = net(SA_image)
            softOutput  = torch.softmax(output, dim = 1).cpu().detach().numpy()     # (10, 4, 256, 256)

        if 'AMBW' in self.experiment_name:
            outputs_tanh1, outputs1, outputs_tanh2, outputs2 = net(SA_image)
            softOutput  = (torch.softmax(outputs1, dim = 1) + torch.softmax(outputs2, dim = 1)) / 2
            softOutput  = softOutput.cpu().detach().numpy()                         # (10, 4, 256, 256)

        if 'E1D2_FS' in self.experiment_name:
            outputs_tanh1, outputs1, outputs_tanh2, outputs2 = net(SA_image)
            softOutput  = (torch.softmax(outputs1, dim = 1) + torch.softmax(outputs2, dim = 1)) / 2
            softOutput  = softOutput.cpu().detach().numpy()                        # (10, 4, 256, 256)

        softOutput      = rearrange_dims(softOutput)                               # (4, 256, 256, 10)

        #----------------------------------------------------------------------#
        # Confirmation that the generated softOutputResamp is indeed a
        # probability tensor i.e., it should sum to one across class dimension.
        print('sum across classes (before post-processing):',
               np.max(np.sum(softOutput, axis = 0)), np.min(np.sum(softOutput, axis = 0)))

        #----------------------------------------------------------------------#
        sa_seg  = np.argmax(softOutput, axis = 0)
        sa_seg  = sa_seg.astype(np.uint8)
        sa_seg  = zoom(sa_seg, (sa_ed_shape[0]/self.crop_factor[0], sa_ed_shape[1]/self.crop_factor[1], 1.0), order=0)

        if 'dtc' in self.experiment_name:
            return sa_seg, softOutput, out_tanh
        else:
            return sa_seg, softOutput

    #------------------------------------------------------------------------------#
    def get_dice_hd95(self):

        for case in os.listdir(self.input_folder):
            # Loading SA images for each subject
            sa_ed = nib.load(os.path.join(self.input_folder, case, case + '_SA_ED_gt.nii.gz'))      # Input_dir/161/161_SA_ED.nii.gz
            sa_es = nib.load(os.path.join(self.input_folder, case, case + '_SA_ES_gt.nii.gz'))      # Input_dir/161/161_SA_ES.nii.gz

            # Loading groudtruth segmentations
            sa_ed_affine, sa_ed_header, sa_ed_pix_size, sa_ed_shape, sa_ed_gt = read_nifti_image(sa_ed, islabel=True)
            sa_es_affine, sa_es_header, sa_es_pix_size, sa_es_shape, sa_es_gt = read_nifti_image(sa_es, islabel=True)

            if self.experiment_name == 'dtc':
                if self.save_lsf:
                    self.save_sdf(sa_ed, case, image = 'ed')
                    self.save_sdf(sa_es, case, image = 'es')

            sa_ed_pred_path = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder, self.labeled_ratio,
                                                                      self.experiment_name, case, 'SA_ED_pred')
            sa_es_pred_path = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder, self.labeled_ratio,
                                                                      self.experiment_name, case, 'SA_ES_pred')

            _, _, _, _, sa_ed_pred = read_nifti_image(sa_ed_pred_path, islabel=True)
            _, _, _, _, sa_es_pred = read_nifti_image(sa_es_pred_path, islabel=True)
            print ('sa_ed_pred_path = ', sa_ed_pred_path, sa_ed_pred.shape)

            metrics1    = np.array(test_single_volume(sa_ed_pred, sa_ed_gt, self.num_classes,
                                                      sa_ed_pix_size, sa_ed_shape))
            metrics2    = np.array(test_single_volume(sa_es_pred, sa_es_gt, self.num_classes,
                                                      sa_es_pix_size, sa_es_shape))

            metrics     = np.mean(np.stack([metrics1, metrics2], axis=0), axis=0)
            mean_metric = np.mean(metrics, axis = 0)

            if self.get_FOMs:
                print ('mean_dice = ', mean_metric[0])
                print ('mean hd95 = ', mean_metric[1])
                print ('mean assd = ', mean_metric[2])

            # log scores onto csv file
            csv_logger.writerow([case,                   # patient id
                                 metrics[0][0],          # LV dice
                                 metrics[1][0],          # MYO dice
                                 metrics[2][0],          # RV dice
                                 metrics[0][1],          # LV hd95
                                 metrics[1][1],          # MYO hd95
                                 metrics[2][1],          # RV hd95
                                 metrics[0][2],          # LV assd
                                 metrics[1][2],          # MYO assd
                                 metrics[2][2]])         # RV assd
            csvfile.flush()

        return metrics

    #------------------------------------------------------------------------------#
    def save_sdf(self, gt_image, case, image='es'):
        gt_dis  = []
        temp_gt = torch.Tensor(gt_image.get_fdata().astype(np.uint8)).permute(2,0,1)
        with torch.no_grad():
            for i in range(0,4):
                temp = compute_sdf(temp_gt.cpu().numpy() == i, temp_gt.shape)
                gt_dis.append(temp)

        gt_dis_image = torch.Tensor(np.array(gt_dis)).permute(0,2,3,1).numpy()

        if image == 'ed':
            sa_ed_affine            = gt_image.affine
            sa_ed_header            = gt_image.header
            sa_ed_pred_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder, self.labeled_ratio,
                                                                              self.experiment_name, case, 'SA_ED_gt_lsf')
            sa_ed_pred              = nib.Nifti1Image(gt_dis_image, sa_ed_affine, sa_ed_header)
            nib.save(sa_ed_pred, sa_ed_pred_path)

        else:
            sa_es_affine            = gt_image.affine
            sa_es_header            = gt_image.header
            sa_es_pred_path         = '{0}/{1}/{2}/{3}/{3}_{4}.nii.gz'.format(self.output_folder, self.labeled_ratio,
                                                                              self.experiment_name, case, 'SA_ES_gt_lsf')
            sa_es_pred              = nib.Nifti1Image(gt_dis_image, sa_es_affine, sa_es_header)
            nib.save(sa_es_pred, sa_es_pred_path)

#------------------------------------------------------------------------------#
if __name__== '__main__':

    import argparse

    parser          = argparse.ArgumentParser(description = 'inference over MM2 dataset')
    parser.add_argument('--config', type = str, required = True, help = '.yaml config file')
    args            = parser.parse_args()

    inference_obj   = inference_routine(config_file = args.config)

    output_folder   = inference_obj.output_folder
    labeled_ratio   = inference_obj.labeled_ratio
    experiment_name = inference_obj.experiment_name

    snapshot_path = "{}/{}/{}".format(inference_obj.output_folder, labeled_ratio, experiment_name)  # challenge_outputs/0.2/fully_supervised

    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

    inference_obj.predict()

    if inference_obj.save_FOMs_report:
        with open(snapshot_path + '/inference_results.csv', 'a') as csvfile:
            csv_logger = csv.writer(csvfile)
            csv_logger.writerow(['id', 'lv_dice', 'myo_dice', 'rv_dice',
                                       'lv_hd95', 'myo_hd95', 'rv_hd95',
                                       'lv_assd', 'myo_assd', 'rv_assd'])
            inference_obj.get_dice_hd95()

# ---------------------------------------------------------------------------- #

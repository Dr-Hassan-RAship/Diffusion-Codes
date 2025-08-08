#------------------------------------------------------------------------------#
# File name         : train_fully_supervised.py
# Purpose           : Train U-Net fully supervised on the Kvasir dataset
# Usage (command)   : python train_fully_supervised.py --config config_fully_supervised.yaml
#                     tensorboard --logdir='snapshot/fully_supervised'
#
# Authors           : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email             : hassan.mohyuddin@lums.edu.pk
#
# Last Date         : June 11, 2025
#------------------------------------------------------------------------------#

import torch, argparse, logging, os, random, shutil, sys, time, csv, time

import numpy                        as np
import torch.nn                     as nn
import torch.nn.functional          as F
import torch.optim                  as optim
import torch.backends.cudnn         as cudnn
import segmentation_models_pytorch  as smp

from torch.utils.data               import DataLoader
from torch.nn.modules.loss          import CrossEntropyLoss
from monai.metrics                  import DiceMetric, MeanIoU
from medpy                          import metric

from tqdm                           import tqdm
from torchinfo                      import summary
from torchvision                    import transforms
from tensorboardX                   import SummaryWriter

from UM_Net                         import *
from dataloaders.dataset            import *

from helpers.train_utils            import *
from helpers.losses                 import DiceLoss
from helpers.parse_yaml             import parse_yaml_config
from helpers.fom                    import iou_on_batch, dice_coef, DiceLoss

#------------------------------------------------------------------------------#
class TrainSession:
    """class for managing training sessions"""

    def __init__(self, config = None, config_file = None):

        print('---------------------------------')
        print('> Initializing Training Session <')
        print('---------------------------------')

        if config is None:
            assert config_file is not None, f"`config_file` is needed if `config` not provided."
            assert os.path.exists(config_file), f"config file {config_file} not found."
            config = parse_yaml_config(config_file)

        #-------------------------------------------------------------#
        config_data              = config['data']

        self.train_root_path     = config_data.get('train_root_path')
        self.val_root_path       = config_data.get('val_root_path')
        self.experiment_name     = config_data.get('experiment_name')
        self.num_classes         = config_data.get('num_classes')
        self.input_channels      = config_data.get('input_channels')

        #-------------------------------------------------------------#
        config_net              = config['network']

        self.slice_size         = config_net.get('slice_size')
        self.initialize         = config_net.get('initialization')
        self.fix_seed           = config_net.get('fix_seed')
        self.seed               = config_net.get('seed')
        self.batch_size         = config_net.get('batch_size')
        self.epochs             = config_net.get('epochs')
        self.num_workers        = config_net.get('num_workers')
        self.deterministic      = config_net.get('deterministic')

        #-------------------------------------------------------------#
        config_optim            = config['optimization']

        self.optimizer          = config_optim.get('optimizer')
        self.LR_policy          = config_optim.get('LR_policy')
        self.eta_zero           = config_optim.get('eta_zero')
        self.eta_N              = config_optim.get('eta_N')
        self.lr_decay_rate      = config_optim.get('LR_decay_rate')

        #-------------------------------------------------------------#
        print ('Configuration Parameters = ', config_file, config_net, config_optim)

    #------------------------------------------------------------------------------#
    def train(self, snapshot_path):

        num_classes     = self.num_classes
        batch_size      = self.batch_size
        epochs          = self.epochs

        model           = UM_Net(num_classes = self.num_classes).cuda()

        # -------------------------------------------------------------------- #
        # Training data
        db_train        = Polyp_Dataset(root        = self.train_root_path,
                                        mode        = "train",
                                        slice_size  = self.slice_size)
        # Validation data
        db_val          = Polyp_Dataset(root        = self.val_root_path,
                                        mode        = "val",
                                        slice_size  = self.slice_size)

        def worker_init_fn(worker_id): random.seed(self.seed + worker_id)

        trainloader     = DataLoader(db_train,
                                     batch_size     = batch_size,
                                     shuffle        = True,
                                     num_workers    = self.num_workers,
                                     pin_memory     = True,
                                     worker_init_fn = worker_init_fn)

        valloader       = DataLoader(db_val,
                                     batch_size     = batch_size,
                                     shuffle        = False,
                                     num_workers    = self.num_workers)

        # -------------------------------------------------------------------- #
        # Define optimizer and Losses
        optimizer       = optim_policy(self, model)
        if self.LR_policy == "StepDecay": lrVal = self.eta_zero
        else: lrVal     = []

        # ce_loss         = CrossEntropyLoss()
        bce_loss        = smp.losses.SoftBCEWithLogitsLoss()
        # bce_loss        = nn.BCELoss()
        dice_loss       = smp.losses.DiceLoss(mode = 'binary', from_logits = True) # DiceLoss(num_classes)
        # dice_loss       = DiceLoss(num_classes)

        # -------------------------------------------------------------------- #
        writer          = SummaryWriter(snapshot_path + '/log')
        logging.info("{} iterations per epoch".format(len(trainloader)))

        iter_num                        = 0
        best_performance, best_epoch    = 0.0, 0
        iterator                        = tqdm(range(epochs), ncols = 70)
        self.total_iterations           = len(trainloader) * epochs
        print ('Total # of iterations = ', self.total_iterations)

        for epoch_num in iterator:
            train_loss, train_dice_scores, train_mIoU_scores = [], [], []

            # ******************* Training ******************* #
            torch.cuda.empty_cache(); model.train()
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                out1, out2, out3, out4, out5    = model(volume_batch)       # volume batch passed through model
                out_soft1                       = torch.sigmoid(out1)      # compute softmax prob. maps
                out_soft2                       = torch.sigmoid(out2)
                out_soft3                       = torch.sigmoid(out3)
                out_soft4                       = torch.sigmoid(out4)
                out_soft5                       = torch.sigmoid(out5)
                out_soft_avg                    = (1/5)* (out_soft1 + out_soft2 + out_soft3 + out_soft4 + out_soft5)

                var1, omega1    = calculate_discrepancy(out_soft1, label_batch)
                var2, omega2    = calculate_discrepancy(out_soft2, label_batch)
                var3, omega3    = calculate_discrepancy(out_soft3, label_batch)
                var4, omega4    = calculate_discrepancy(out_soft4, label_batch)
                var5, omega5    = calculate_discrepancy(out_soft5, label_batch)

                omega_sum       = omega1 + omega2 + omega3 + omega4 + omega5

                loss_var        = (1/5) * (var1 + var2 + var3 + var4 + var5)

                #pred         = (outputs_soft > 0.5).detach()
                #outputs_soft = torch.sigmoid(outputs)   # compute softmax prob. maps

                train_dice_scores.append(dice_coef(out_soft_avg[:,0,:,:], label_batch[:,0,:,:]))
                # train_mIoU_scores.append(iou_on_batch(pred, label_batch[:,0,:,:]))

                loss_ce         = bce_loss(out1, label_batch.float())
                loss_dice       = dice_loss(out1, label_batch.float())
                loss_1          = 0.5 * (loss_dice + loss_ce)

                loss_ce_2       = bce_loss(out2, label_batch.float())
                loss_dice_2     = dice_loss(out2, label_batch.float())
                loss_2          = 0.5 * (loss_dice_2 + loss_ce_2)

                loss_ce_3       = bce_loss(out3, label_batch.float())
                loss_dice_3     = dice_loss(out3, label_batch.float())
                loss_3          = 0.5 * (loss_dice_3 + loss_ce_3)

                loss_ce_4       = bce_loss(out4, label_batch.float())
                loss_dice_4     = dice_loss(out4, label_batch.float())
                loss_4          = 0.5 * (loss_dice_4 + loss_ce_4)

                loss_ce_5       = bce_loss(out5, label_batch.float())
                loss_dice_5     = dice_loss(out5, label_batch.float())
                loss_5          = 0.5 * (loss_dice_5 + loss_ce_5)

                #loss            = (omega1 * loss_1 + omega2 * loss_2 + omega3 * loss_3 + \
                #omega4 * loss_4 + omega5 * loss_5) / omega_sum # + loss_var
                loss = (1/5) * (loss_1 + loss_2 + loss_3 + loss_4 + loss_5)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_loss.append(loss.cpu().detach().numpy())

                # Increment the iteration number
                iter_num     = iter_num + 1

                # Log the learning rate and losses againt the iteration number for Tensorboard plots
                writer.add_scalar('Train_Loss/total_loss'   , loss      , iter_num)
                writer.add_scalar('Train_Loss/loss_ce'      , loss_ce   , iter_num)
                writer.add_scalar('Train_Loss/loss_dice'    , loss_dice , iter_num)
                writer.add_scalar('Train_Loss/loss_var'     ,  loss_var , iter_num)

                # log loss and dice scores on command window
                logging.info(
                    'Batch %d : loss : %f, loss_ce: %f, loss_dice: %f, dice_score: %f' %
                    (i_batch, loss.item(), loss_ce.item(), loss_dice.item(),
                     train_dice_scores[-1]))

            # ---------------------------------------------------------------- #
            # Varying learning rate by epoch
            param_group, optimizer, lrVal  = LR_schedule(self, optimizer, iter_num, lrVal)
            writer.add_scalar('Learning_Rate/lr', lrVal, iter_num)
            # ---------------------------------------------------------------- #

            # ******************* Validation BEGINS ******************* #
            torch.cuda.empty_cache(); model.eval()
            val_loss, val_loss_ce, val_loss_dice, val_dice_scores, val_mIoU_scores = [], [], [], [], []

            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                    out1, out2, out3, out4, out5    = model(volume_batch)       # volume batch passed through model
                    out_soft1                       = torch.sigmoid(out1)      # compute softmax prob. maps
                    out_soft2                       = torch.sigmoid(out2)
                    out_soft3                       = torch.sigmoid(out3)
                    out_soft4                       = torch.sigmoid(out4)
                    out_soft5                       = torch.sigmoid(out5)
                    out_soft_avg                    = (1/5)* (out_soft1 + out_soft2 + out_soft3 + out_soft4 + out_soft5)

                    #pred         = (outputs_soft > 0.5).detach()
                    #outputs_soft = torch.sigmoid(outputs)   # compute softmax prob. maps

                    val_dice_scores.append(dice_coef(out_soft_avg[:,0,:,:], label_batch[:,0,:,:]))
                    # train_mIoU_scores.append(iou_on_batch(pred, label_batch[:,0,:,:]))

                    loss_ce         = bce_loss(out1, label_batch.float())
                    loss_dice       = dice_loss(out1, label_batch.float())
                    loss_1          = 0.5 * (loss_dice + loss_ce)

                    loss_ce_2       = bce_loss(out2, label_batch.float())
                    loss_dice_2     = dice_loss(out2, label_batch.float())
                    loss_2          = 0.5 * (loss_dice_2 + loss_ce_2)

                    loss_ce_3       = bce_loss(out3, label_batch.float())
                    loss_dice_3     = dice_loss(out3, label_batch.float())
                    loss_3          = 0.5 * (loss_dice_3 + loss_ce_3)

                    loss_ce_4       = bce_loss(out4, label_batch.float())
                    loss_dice_4     = dice_loss(out4, label_batch.float())
                    loss_4          = 0.5 * (loss_dice_4 + loss_ce_4)

                    loss_ce_5       = bce_loss(out5, label_batch.float())
                    loss_dice_5     = dice_loss(out5, label_batch.float())
                    loss_5          = 0.5 * (loss_dice_5 + loss_ce_5)

                    loss            = 0.5 * (loss_1 + loss_2 + loss_3 + loss_4 + loss_5)

                    # outputs      = model(volume_batch)
                    # outputs_soft = torch.sigmoid(outputs)
                    #
                    # val_dice_scores.append(dice_coef(outputs_soft[:, 0, :, :], label_batch[:, 0, :, :]))
                    # # val_mIoU_scores.append(iou_on_batch(pred, label_batch[:, 1, :, :]))
                    #
                    # loss_ce     = bce_loss(outputs, label_batch.float())
                    # loss_dice   = dice_loss(outputs, label_batch.float())
                    # loss        = 0.5 * (loss_dice + loss_ce)

                    val_loss.append(loss.cpu().detach().numpy())
                    val_loss_ce.append(loss_ce.cpu().detach().numpy())
                    val_loss_dice.append(loss_dice.cpu().detach().numpy())

            # Losses
            mean_train_loss     = np.mean(np.array(train_loss))
            mean_val_loss       = np.mean(np.array(val_loss))
            mean_val_loss_ce    = np.mean(np.array(val_loss_ce))
            mean_val_loss_dice  = np.mean(np.array(val_loss_dice))

            # log mean train, validation loss on command window
            logging.info('\nMean Training Loss      : %f'   % (mean_train_loss))
            logging.info('\nMean Validation Loss    : %f'   % (mean_val_loss))

            # write losses to tensorboard
            writer.add_scalar('Val_Loss/validation_loss', mean_val_loss     , epoch_num)
            writer.add_scalar('Val_Loss/CE_loss'        , mean_val_loss_ce  , epoch_num)
            writer.add_scalar('Val_Loss/dice_loss'      , mean_val_loss_dice, epoch_num)
            writer.add_scalars('Loss', {'training': mean_train_loss, 'validation': mean_val_loss}, epoch_num)

            # validation mean dice and mIoU scores over batches
            performance     = np.mean(np.array(val_dice_scores), axis = 0)
            # val_mIoU        = np.mean(np.array(val_mIoU_scores), axis = 0)

            # train mean dice scores over batches
            train_dice      = np.mean(np.array(train_dice_scores), axis = 0)
            # train_mIoU      = np.mean(np.array(train_mIoU_scores), axis = 0)

            # write dice scores for each class to tensorboard
            writer.add_scalar('Validation/val_mean_dice'        , performance       , epoch_num)
            # writer.add_scalar('Validation/val_mean_mIoU'        , val_mIoU          , epoch_num)
            writer.add_scalar('Validation/Best_val_mean_dice'   , best_performance  , epoch_num)

            if performance > best_performance:  # if dice score improves
                best_epoch          = epoch_num
                best_performance    = performance
                save_best           = os.path.join(snapshot_path, 'models', 'best_model.pth')
                torch.save(model.state_dict(), save_best)
                writer.add_scalar('Validation/Best_val_mean_dice', best_performance , epoch_num)

            save_mode_path  = os.path.join(snapshot_path, 'models', 'epoch_num_{}.pth'.format(epoch_num))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            logging.info('Mean dice_score: %f' % (performance))

            # log scores onto csv file
            csv_logger.writerow([epoch_num,                         # Epoch
                                 train_dice,                        # Train_Dice
                                 mean_train_loss,                   # Mean_Train_Loss
                                 performance,                       # Val_Dice
                                 mean_val_loss])                    # mean_val_loss

            logging.info('\nBest Epoch : %d with Best Performance : %f \n' % (best_epoch, best_performance))

            csvfile.flush()
            model.train(); torch.cuda.empty_cache()

            # ******************* Validation ENDS ******************* #
            if epoch_num >= epochs: iterator.close(); break

            # ******************* EARLY stopping ******************* #
            # if self.early_stop:
            #    early_stopping_count = epoch_num - best_epoch
                #logging.info('\n Early stopping count: {}/{}'.format(early_stopping_count, self.patience_interval))
            #    if early_stopping_count > self.patience_interval:
            #        logging.info('\n Early stopping!')
            #        break

            # -----------------------------------------------------------------#
            writer.close()

        print('Best performance: ', best_epoch, best_performance)
        return "Training Finished!"

################################################################################
if __name__ == "__main__":
    # get the start time
    st      = time.time()

    parser  = argparse.ArgumentParser(description = 'Train fully supervised U-Net')
    parser.add_argument('--config', type = str, required = True, help = '.yaml config file')
    args    = parser.parse_args()
    sess    = TrainSession(config_file = args.config)

    if sess.fix_seed:
        print("Seed is being fixed")
        os.environ['PYTHONHASHSEED']    = str(sess.seed)
        os.environ['PL_GLOBAL_SEED']    = str(sess.seed)
        os.environ['PL_SEED_WORKERS']   = str(sess.num_workers)
        random.seed(sess.seed)
        np.random.seed(sess.seed)
        np.random.default_rng(sess.seed)
        torch.manual_seed(sess.seed)
        torch.cuda.manual_seed(sess.seed)
        torch.cuda.manual_seed_all(sess.seed)

    if not sess.deterministic:
        cudnn.benchmark     = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark     = False
        cudnn.deterministic = True

    # model directory path is defined
    snapshot_path   = "snapshot/{}".format(sess.experiment_name)

    # Check if path exists else make the directory
    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

    model_path      = os.makedirs(os.path.join(snapshot_path, 'models'), exist_ok = True)

    code_path       = os.path.join(snapshot_path, 'code')
    if not os.path.exists(code_path): os.makedirs(code_path)

    shutil.copy(sys.argv[0], snapshot_path + '/code')
    shutil.copy(sys.argv[2], snapshot_path + '/code')
    logging.basicConfig(filename    =   snapshot_path + "/log.txt",
                        level       =   logging.INFO,                               # This means that only messages with a level of INFO or higher will be logged.
                        format      =   '[%(asctime)s.%(msecs)03d] %(message)s',    # format of time
                        datefmt     =   '%H:%M:%S')                                 # format of date

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(sess))

    with open(snapshot_path + '/logs.csv', 'a') as csvfile:
        csv_logger = csv.writer(csvfile)
        csv_logger.writerow(['Epoch', 'Train DSC'   , 'mean_train_loss',
                                      'Val_DSC'     , 'mean_val_loss'])
        sess.train(snapshot_path)

    # get the end time
    et          = time.time()

    # get the execution time
    res         = et - st; final_res = res / 60
    print('Execution time:', final_res, 'minutes')

################################################################################

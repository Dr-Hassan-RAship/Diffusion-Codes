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
from helpers.inference_utils        import performance_in_training

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
            train_loss, train_dice_scores = [], []

            # ******************* Training ******************* #
            torch.cuda.empty_cache(); model.train()
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                outputs      = model(volume_batch)       # volume batch passed through model
                outputs_soft = torch.sigmoid(outputs)    # compute softmax prob. maps
                #outputs_soft = torch.sigmoid(outputs)   # compute softmax prob. maps

                train_dice_scores.append(performance_in_training(outputs_soft,
                                                                 label_batch,
                                                                 self.num_classes, 'Training'))

                loss_ce      = bce_loss(outputs, label_batch.float())
                loss_dice    = dice_loss(outputs, label_batch.float())
                loss         = 0.5 * (loss_dice + loss_ce)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_loss.append(loss.cpu().detach().numpy())

                # Increment the iteration number
                iter_num     = iter_num + 1

                # Log the learning rate and losses againt the iteration number for Tensorboard plots
                writer.add_scalar('Train_Loss/total_loss'   , loss      , iter_num)
                writer.add_scalar('Train_Loss/loss_ce'      , loss_ce   , iter_num)
                writer.add_scalar('Train_Loss/loss_dice'    , loss_dice , iter_num)

                # log loss and dice scores on command window
                logging.info(
                    'Batch %d : loss : %f, loss_ce: %f, loss_dice: %f, train_DSC: %f' %
                    (i_batch, loss.item(), loss_ce.item(), loss_dice.item(), train_dice_scores[-1][0]))

            # ---------------------------------------------------------------- #
            # Varying learning rate by epoch
            param_group, optimizer, lrVal  = LR_schedule(self, optimizer, iter_num, lrVal)
            writer.add_scalar('Learning_Rate/lr', lrVal, iter_num)
            # ---------------------------------------------------------------- #

            # ******************* Validation BEGINS ******************* #
            torch.cuda.empty_cache(); model.eval()
            val_loss, val_loss_ce, val_loss_dice, val_dice_scores = [], [], [], []

            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                    outputs     = model(volume_batch)
                    outputs_soft= torch.sigmoid(outputs)

                    metric_i    = performance_in_training(outputs_soft, label_batch.unsqueeze(1), self.num_classes, 'Training')
                    val_dice_scores.append(metric_i)

                    loss_ce     = bce_loss(outputs_soft    , label_batch[:].float())
                    loss_dice   = dice_loss(outputs_soft   , label_batch[:].long())
                    loss        = loss_ce #0.5 * (loss_dice + loss_ce)

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

            # train mean dice scores and validation mean dice scores over slices
            train_metrics       = np.mean(np.array(train_dice_scores)   , axis = 0)
            val_metrics         = np.mean(np.array(val_dice_scores)     , axis = 0)

            # write dice scores for each class to tensorboard
            for class_i in range(num_classes):
                writer.add_scalar('Validation/val_{}_dice'.format(class_i+1), val_metrics[class_i], iter_num)

            performance         = np.mean(val_metrics)      # mean dice score over classes
            writer.add_scalar('Validation/val_mean_dice', performance, iter_num)

            if performance > best_performance:          # if dice score improves
                best_epoch      = epoch_num
                best_performance= performance
                save_best       = os.path.join(snapshot_path, 'models', 'best_model.pth')
                torch.save(model.state_dict(), save_best)

            save_mode_path      = os.path.join(snapshot_path, 'models', 'epoch_num_{}.pth'.format(epoch_num))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            # Print scores on the command window
            logging.info('\nmean_dice : %f \n'  % (val_metrics[0]))

            # log scores onto csv file
            csv_logger.writerow([epoch_num,                         # Epoch
                                 train_metrics[0],                  # mean_train_dice
                                 mean_train_loss,                   # mean_train_loss
                                 val_metrics[0],                    # mean_val_dice
                                 mean_val_loss])                    # mean_val_loss

            logging.info('\nBest Epoch : %d with Best Performance : %f \n' % (best_epoch, best_performance))

            csvfile.flush()
            model.train(); torch.cuda.empty_cache()

            # ******************* Validation ENDS ******************* #
            if epoch_num >= epochs: iterator.close(); break

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

    model_path      = os.makedirs(os.path.join(snapshot_path, 'models'))

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

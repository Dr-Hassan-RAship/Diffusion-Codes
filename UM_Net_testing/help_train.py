#------------------------------------------------------------------------------#
#
# File name         : train_fully_supervised.py
# Purpose           : Train a U-Net model (ResNet encoder) on QatCov-19 Dataset
# Usage (command)   : python train_fully_supervised.py --config config_fully_supervised.yaml
#                     tensorboard --logdir='snapshot/FS_100p'
#                     tensorboard --logdir='snapshot/LS_100/'
#
# Authors           : Shujah Ur Rehman, Syed Muqeem Mahmood, Hassan Mohy-ud-Din
# Email             : 21060003@lums.edu.pk, 24100025@lums.edu.pk,
#                     hassan.mohyuddin@lums.edu.pk
#
# Last Date         : October 15, 2024
#
#------------------------------------------------------------------------------#

import torch, argparse, logging, os, random, shutil, sys, time, csv, time

import numpy                        as np
import torch.backends.cudnn         as cudnn

from tensorboardX                   import SummaryWriter
from tqdm                           import tqdm
from torchinfo                      import summary
from torch.utils.data               import DataLoader
from torchvision                    import transforms
from torch.nn.modules.loss          import CrossEntropyLoss

from networks.net_factory           import net_factory
from dataloaders.dataset            import ImageToImage2D, RandomGenerator, ValGenerator

from helpers.train_utils            import *
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

        self.seed               = config_net.get('seed')
        self.fix_seed           = config_net.get('fix_seed')
        self.net_model          = config_net.get('net_model')
        self.slice_size         = config_net.get('slice_size')
        self.batch_size         = config_net.get('batch_size')
        self.epochs             = config_net.get('epochs')
        self.early_stop         = config_net.get('early_stop')
        self.patience_interval  = config_net.get('patience_interval')
        self.deterministic      = config_net.get('deterministic')

        #-------------------------------------------------------------#
        config_optim            = config['optimization']

        self.optimizer          = config_optim.get('optimizer')
        self.LR_policy          = config_optim.get('LR_policy')
        self.eta_zero           = config_optim.get('eta_zero')
        self.eta_N              = config_optim.get('eta_N')
        self.eta_min            = config_optim.get('eta_min')
        self.lr_decay_rate      = config_optim.get('LR_decay_rate')
        self.lr_by_iter         = config_optim.get('LR_by_iter')

        #-------------------------------------------------------------#
        print ('Configuration Parameters = ', config_file, config_net, config_optim)

    #------------------------------------------------------------------------------#
    def train(self, snapshot_path):

        input_channels  = self.input_channels
        num_classes     = self.num_classes
        batch_size      = self.batch_size
        epochs          = self.epochs

        model           = net_factory(net_type      = self.net_model,
                                      in_chns       = input_channels,
                                      class_num     = num_classes,
                                      pretrain      = True,
                                      concatF       = True,
                                      init          = None).cuda()

        summary(model, input_size = (batch_size, input_channels, self.slice_size[0], self.slice_size[1]))

        # -------------------------------------------------------------------- #
        # Training data
        db_train        = ImageToImage2D(base_dir     = self.train_root_path,
                                         split        = 'train_lab',
                                         transform    = transforms.Compose([RandomGenerator(self.slice_size)]),
                                         one_hot_mask = True,
                                         image_size   = self.slice_size)

        # Validation data
        db_val          = ImageToImage2D(base_dir     = self.val_root_path,
                                         split        = 'val',
                                         transform    = transforms.Compose([ValGenerator(self.slice_size)]),
                                         one_hot_mask = True,
                                         image_size   = self.slice_size)

        def worker_init_fn(worker_id): random.seed(self.seed + worker_id)

        trainloader     = DataLoader(db_train,
                                     batch_size     = batch_size,
                                     shuffle        = True,
                                     num_workers    = 6,
                                     pin_memory     = True,
                                     worker_init_fn = worker_init_fn)

        valloader       = DataLoader(db_val,
                                     batch_size   = batch_size,
                                     shuffle      = False,
                                     num_workers  = 6)
        model.train()

        # -------------------------------------------------------------------- #
        max_iters       = len(trainloader) * epochs
        print ('Total # of iterations = ', max_iters)

        # -------------------------------------------------------------------- #
        # Define optimizer and Losses
        optimizer       = optim_policy(self, model)

        if self.LR_policy == "StepDecay": lrVal = self.eta_zero
        else: lrVal     = []

        dice_loss       = DiceLoss(num_classes)
        ce_loss         = CrossEntropyLoss()
        # -------------------------------------------------------------------- #
        writer          = SummaryWriter(snapshot_path + '/log')
        logging.info("{} iterations per epoch".format(len(trainloader)))

        iter_num                        = 0
        best_performance, best_epoch    = float('-inf'), 0
        iterator                        = tqdm(range(epochs), ncols = 70)
        self.total_iterations           = len(trainloader) * epochs

        for epoch_num in iterator:
            train_loss, train_dice_scores, train_mIoU_scores = [], [], []

            torch.cuda.empty_cache()
            # ******************* Training ******************* #
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch   = sampled_batch['image'], sampled_batch['mask']
                volume_batch, label_batch   = volume_batch.cuda(), label_batch.cuda()

                outputs                     = model(volume_batch)
                outputs_soft                = torch.softmax(outputs, dim = 1)
                pred                        = torch.argmax(outputs_soft, dim=1).detach()

                train_dice_scores.append(dice_coef(outputs_soft[:,1,:,:], label_batch[:,1,:,:]))
                train_mIoU_scores.append(iou_on_batch(pred, label_batch[:,1,:,:]))

                loss_ce                     = ce_loss(outputs, label_batch[:].float())
                loss_dice                   = dice_loss(outputs_soft, label_batch)
                loss                        = 0.5 * (loss_dice + loss_ce)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_loss.append(loss.cpu().detach().numpy())

                # Increment the iteration number
                iter_num                    = iter_num + 1

                # Log the learning rate and losses againt the iteration number for Tensorboard plots
                writer.add_scalar('Train_Loss/total_loss'   , loss                  , iter_num)
                writer.add_scalar('Train_Loss/loss_ce'      , loss_ce               , iter_num)
                writer.add_scalar('Train_Loss/loss_dice'    , loss_dice             , iter_num)
                writer.add_scalar('Train_mIoU/train_mIoU'   , train_mIoU_scores[-1] , iter_num)

                # log loss and scores on command window
                logging.info(
                    'Batch %d : loss : %f, loss_ce: %f, loss_dice: %f, dice_score: %f, mIoU_score: %f' %
                    (i_batch, loss.item(), loss_ce.item(), loss_dice.item(),
                     train_dice_scores[-1], train_mIoU_scores[-1]))

                # Varying learning rate by iteration (if the flag is True)
                if self.lr_by_iter:
                    if self.LR_policy != "StepDecay": self.LR_policy = "PolyDecay"
                    param_group, optimizer, lrVal = LR_schedule(self, optimizer, iter_num, lrVal)
                    writer.add_scalar('Learning_Rate/lr', lrVal, iter_num)

                # ---------------------------------------------------------- #

            # Varying learning rate by epoch (if the lr_by_iter flag is False)
            if self.lr_by_iter == False:
                param_group, optimizer, lrVal = LR_schedule(self, optimizer, iter_num, lrVal)
                writer.add_scalar('Learning_Rate/lr', lrVal, iter_num)

            # ******************* Validation BEGINS ******************* #
            torch.cuda.empty_cache(); model.eval()

            # Validation Losses
            val_loss, val_loss_ce, val_loss_dice    = [], [], []
            val_dice_scores, val_mIoU_scores        = [], []

            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_batch, label_batch   = sampled_batch['image'], sampled_batch['mask']
                    volume_batch, label_batch   = volume_batch.cuda(), label_batch.cuda()

                    outputs                     = model(volume_batch)
                    outputs_soft                = torch.softmax(outputs, dim = 1)
                    pred                        = torch.argmax(outputs_soft, dim=1).detach()

                    val_dice_scores.append(dice_coef(outputs_soft[:,1,:,:], label_batch[:,1,:,:]))
                    val_mIoU_scores.append(iou_on_batch(pred, label_batch[:,1,:,:]))

                    loss_ce                     = ce_loss(outputs, label_batch.float())
                    loss_dice                   = dice_loss(outputs_soft, label_batch.long())
                    loss                        = 0.5 * (loss_ce + loss_dice)

                    val_loss_ce.append(loss_ce.cpu().detach().numpy())
                    val_loss_dice.append(loss_dice.cpu().detach().numpy())
                    val_loss.append(loss.cpu().detach().numpy())

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
            val_mIoU        = np.mean(np.array(val_mIoU_scores), axis = 0)

            # train mean dice scores over batches
            train_dice      = np.mean(np.array(train_dice_scores), axis = 0)
            train_mIoU      = np.mean(np.array(train_mIoU_scores), axis = 0)

            # write dice scores for each class to tensorboard
            writer.add_scalar('Validation/val_mean_dice'        , performance       , epoch_num)
            writer.add_scalar('Validation/val_mean_mIoU'        , val_mIoU          , epoch_num)
            writer.add_scalar('Validation/Best_val_mean_dice'   , best_performance  , epoch_num)

            if performance > best_performance:  # if dice score improves
                best_epoch          = epoch_num
                best_performance    = performance
                save_best           = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_best)
                writer.add_scalar('Validation/Best_val_mean_dice', best_performance , epoch_num)

            save_mode_path  = os.path.join(snapshot_path, 'epoch_num_{}.pth'.format(epoch_num))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            logging.info('Mean dice_score: %f, Mean mIoU_score: %f' % (performance, val_mIoU))

            # log scores onto csv file
            csv_logger.writerow([epoch_num,                         # Epoch
                                 train_dice,                        # Train_Dice
                                 train_mIoU,                        # Train_mIoU
                                 mean_train_loss,                   # Mean_Train_Loss
                                 performance,                       # Val_Dice
                                 val_mIoU,                          # Val_mIoU
                                 mean_val_loss])                    # mean_val_loss

            logging.info('\nBest Epoch : %d with Best Performance : %f \n' % (best_epoch, best_performance))

            csvfile.flush()
            model.train(); torch.cuda.empty_cache()

            # ******************* Validation ENDS ******************* #
            if epoch_num >= epochs: iterator.close(); break

            # ******************* EARLY stopping ******************* #
            if self.early_stop:
                early_stopping_count = epoch_num - best_epoch
                #logging.info('\n Early stopping count: {}/{}'.format(early_stopping_count, self.patience_interval))
                if early_stopping_count > self.patience_interval:
                    logging.info('\n Early stopping!')
                    break

            # -----------------------------------------------------------------#
        writer.close()

        print('Best performance: ', best_epoch, best_performance)
        return "Training Finished!"

################################################################################
if __name__ == "__main__":
    import argparse

    # get the start time
    st                      = time.time()

    parser                  = argparse.ArgumentParser(description = 'Train fully supervised U-Net')
    parser.add_argument('--config', type = str, required = True, help = '.yaml config file')
    args                    = parser.parse_args()
    sess                    = TrainSession(config_file = args.config)

    if sess.fix_seed:
        print("Seed is being fixed")
        os.environ['PYTHONHASHSEED'] = str(sess.seed)
        random.seed(sess.seed)
        np.random.seed(sess.seed)
        np.random.default_rng(sess.seed)
        torch.manual_seed(sess.seed)
        torch.cuda.manual_seed(sess.seed)
        torch.cuda.manual_seed_all(sess.seed)

    # # Optimizing memory fragmentation
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    if not sess.deterministic:
        cudnn.benchmark     = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark     = False
        cudnn.deterministic = True

    # model directory path is defined
    snapshot_path           = "snapshot/{}".format(sess.experiment_name)
    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

    code_path               = os.path.join(snapshot_path, 'code')
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
        csv_logger          = csv.writer(csvfile)
        csv_logger.writerow(['Epoch', 'train_dice'  , 'train_mIoU'  , 'mean_train_loss',
                                      'val_dice'    , 'val_mIoU'    , 'mean_val_loss'])
        sess.train(snapshot_path)

    # get the end time
    et                      = time.time()

    # get the execution time
    res                     = et - st; final_res = res / 60
    print('Execution time:', final_res, 'minutes')

################################################################################
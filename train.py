import argparse
import random
import os
import math
import time
from sqlalchemy import exists
from tqdm import tqdm
import options.options as option
from torch.utils.data import random_split


def main():
    parser = argparse.ArgumentParser(
        description='Train Super Resolution Models')
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to options JSON file.')
    parser.add_argument(
        '-save_folder', type=str, required=True, help='Name of experiments.')
    save_folder = parser.parse_args().save_folder
    opt = option.parse(parser.parse_args().opt, save_folder)
    # import torch after set CUDA_VISIBLE_DEVICES
    import torch
    from solvers import create_solver
    from data import create_dataloader
    from data import create_dataset
    from utils import util

    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("===> Random Seed: [%d]" % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if opt['is_train']:
        # create folders
        if opt['solver']['pretrain'] == 'resume' or 'debug' in opt['name']:
            pass
        else:
            # rename old experiments if exists
            util.mkdir_and_rename(opt['path']['exp_root'])
            util.mkdirs((path for key, path in opt['path'].items(
            ) if key != 'exp_root' and key != 'tb_logger_root'))
            if opt['use_tb_logger']:
                util.mkdir_and_rename(opt['path']['tb_logger_root'])
            option.save(opt)

        print("===> Experimental DIR: [%s]" % opt['path']['exp_root'])

    # create train and val dataloader
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_set, val_set = random_split(train_set, [int(0.9*len(train_set)), len(train_set) - int(0.9*len(train_set))])
            train_loader = create_dataloader(train_set, dataset_opt)
            total_iters = int(opt['solver']['niter'])
            num_epoch = int(opt['solver']['epochs'])

            print('===> Train Dataset: %s   Number of images: [%d]' %
                  (dataset_opt['name'], len(train_set)))
            if train_loader is None:
                raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            # val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset: %s   Number of images: [%d]' %
                  (dataset_opt['name'], len(val_set)))

        else:
            raise NotImplementedError(
                "[Error] Dataset phase [%s] in *.json is not recognized." %
                phase)

    solver = create_solver(opt)

    scale = opt['scale']
    model_name = opt['networks']['which_model'].upper()

    if opt['use_tb_logger']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger_root'])
        print('===> tensorboardX logger created, log to %s' %
              (opt['path']['tb_logger_root']))

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    start_epoch = solver_log['epoch']
    solver.step = solver_log['step']

    print("Method: %s || Scale: %d || Epoch Range: (%d ~ %d)" %
          (model_name, scale, start_epoch, num_epoch))

    # Early stopping
    patience = opt['solver']['patience']
    count_patience = 0
    num_steps = len(train_loader)
    validate_steps = opt['solver']['validate_steps']
    steps = [int(x*(num_steps-1)) for x in validate_steps]

    ####
    for epoch in range(start_epoch, num_epoch + 1):
        print('\n===> Training Epoch: [%d/%d]...  Learning Rate: %f' %
              (epoch, num_epoch, solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        val_loss_dict = {}
        for k in solver_log['records'].keys():
            if 'val_loss' in k:
                val_loss_dict[k[4:]] = []

        for index, batch in enumerate(train_loader):
            solver.step = index
            if solver.step > total_iters:
                break
            solver.feed_data(batch)
            iter_loss = solver.train_step(index)
            batch_size = batch['LR'].size(0)

            if solver.step % opt['logger']['print_freq'] == 0:
                message = time.ctime()
                message += ' <epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, solver.step, solver.get_current_learning_rate())
                for k, v in iter_loss.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    if opt['use_tb_logger']:
                        tb_logger.add_scalar('train_' + k, v, solver.step)
                print(message)

            solver.update_learning_rate()

            if index in steps:
                print('===> Validating at step', index)
                psnr_list = []
                ssim_list = []
                for iter, batch in enumerate(val_loader):
                    solver.feed_data(batch, need_landmark=False)
                    iter_loss = solver.test()
                    for k, v in iter_loss.items():
                        val_loss_dict[k].append(v)

                    # calculate evaluation metrics
                    visuals = solver.get_current_visual()
                    psnr, ssim = util.calc_metrics(
                        visuals['SR'][-1], visuals['HR'], crop_border=scale)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    if 'LR_path' in batch.keys():
                        img_name = os.path.basename(
                            os.path.splitext(batch['LR_path'][0])[0])
                    else:
                        img_name = os.path.basename(
                            os.path.splitext(batch['HR_path'][0])[0])

                for k, v in val_loss_dict.items():
                    solver_log['records']['val_' + k].append(sum(v) / len(v))
                solver_log['records']['psnr'].append(
                    sum(psnr_list) / len(psnr_list))
                solver_log['records']['ssim'].append(
                    sum(ssim_list) / len(ssim_list))
                solver_log['records']['lr'].append(
                    solver.get_current_learning_rate())
                if opt['use_tb_logger']:
                    tb_logger.add_scalar(
                        'val_psnr_mean',
                        sum(psnr_list) / len(psnr_list),
                        global_step=solver.step)
                    tb_logger.add_scalar(
                        'val_ssim_mean',
                        sum(ssim_list) / len(ssim_list),
                        global_step=solver.step)
                    for k, v in val_loss_dict.items():
                        tb_logger.add_scalar(
                            'val_' + k, sum(v) / len(v), global_step=solver.step)

                # record the best epoch
                step_is_best = False
                if solver_log['best_pred'] < (sum(psnr_list) / len(psnr_list)):
                    solver_log['best_pred'] = (sum(psnr_list) / len(psnr_list))
                    step_is_best = True
                    solver_log['best_step'] = epoch
                    solver.save_checkpoint(epoch, True)
                    count_patience = 0
                else:
                    # solver.save_checkpoint(epoch, False)
                    count_patience += 1
                    if count_patience == patience:
                        print("EARLY STOPPING: Got best PSNR =",
                              solver_log['best_pred'], "after {0} epochs".format(epoch))
                        return
                print(
                    "[%s] PSNR: %.2f   SSIM: %.4f    Loss: %.6f   Best PSNR: %.2f in Step: [%d]"
                    % (opt['datasets']['val']['name'], sum(psnr_list) / len(psnr_list),
                        sum(ssim_list) / len(ssim_list),
                        solver_log['records']['val_loss_total'][-1],
                        solver_log['best_pred'], solver_log['best_step']))
                # save log
                solver_log['step'] = epoch
                solver.set_current_log(solver_log)
                solver.save_current_log()

            # if solver.step % opt['solver']['save_freq'] == 0:
            #     solver.save_checkpoint(epoch, False)

                # update lr

    print('===> Finished !')


if __name__ == '__main__':
    main()

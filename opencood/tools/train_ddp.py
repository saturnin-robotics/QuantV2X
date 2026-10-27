import argparse
import contextlib
import io
import json
import os
import statistics
import glob
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def accumulate_losses(sums, counts, loss_dict, suffix=""):
    """Add the scalar entries of a loss dict to the epoch sums."""
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            if v.numel() != 1:
                continue
            v = v.item()
        if isinstance(v, (int, float)):
            sums[k + suffix] = sums.get(k + suffix, 0.0) + float(v)
            counts[k + suffix] = counts.get(k + suffix, 0) + 1


def print_epoch_losses(epoch, total_epochs, sums, counts, distributed):
    """One line per epoch: mean of each loss term over every iteration and
    every rank (replaces the per-iteration print of criterion.logging).
    Dicts are gathered as objects, so ranks may hold different terms."""
    gathered = [(sums, counts)]
    if distributed:
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, (sums, counts))
    total, n = {}, {}
    for s_r, c_r in gathered:
        for k in s_r:
            total[k] = total.get(k, 0.0) + s_r[k]
            n[k] = n.get(k, 0) + c_r[k]
    if not total:
        return
    keys = sorted(total)
    # epochs printed from 1, as the net_epoch<N>.pth checkpoints are named
    print('[epoch %d/%d] train mean over %d it || ' % (epoch + 1, total_epochs, max(n.values()))
          + ' || '.join('%s: %.4f' % (k, total[k] / n[k]) for k in keys),
          flush=True)


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        # CoopFGNN patch: best validation loss of the whole run, kept across
        # resubmitted jobs (it restarted at 1e5 and each job left its own
        # best-val checkpoint behind)
        best_val_path = os.path.join(saved_path, 'best_val.json')
        if os.path.exists(best_val_path):
            with open(best_val_path) as f:
                best_val = json.load(f)
            lowest_val_loss = best_val['loss']
            lowest_val_epoch = best_val['epoch']
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True) # True
        model_without_ddp = model.module


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    # CoopFGNN patch: restore the optimizer state saved with the last epoch
    opt_state_path = os.path.join(saved_path, 'optimizer_last.pth')
    if opt.model_dir and os.path.exists(opt_state_path):
        opt_state = torch.load(opt_state_path, map_location='cpu')
        if opt_state['epoch'] == init_epoch:
            optimizer.load_state_dict(opt_state['optimizer'])
            print('restored optimizer state of epoch %d' % init_epoch)

    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, init_epoch=init_epoch,
        n_iter_per_epoch=len(train_loader))
    per_iter_lr = hypes['lr_scheduler']['core_method'] == 'cosineannealwarm'

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model_without_ddp.model_train_init()
        except:
            print("No model_train_init function")
        loss_sums, loss_counts = {}, {}
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            # tensorboard every iteration, loss printed once per epoch
            with contextlib.redirect_stdout(io.StringIO()):
                criterion.logging(epoch, i, len(train_loader), writer)
            accumulate_losses(loss_sums, loss_counts, criterion.loss_dict)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")
                accumulate_losses(loss_sums, loss_counts, criterion.loss_dict, "_single")

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if per_iter_lr:
                scheduler.step_update(epoch * len(train_loader) + i + 1)


        # torch.cuda.empty_cache() # it will destroy memory buffer
        print_epoch_losses(epoch, epoches, loss_sums, loss_counts, opt.distributed)

        # CoopFGNN patch: only rank 0 writes checkpoints (all ranks wrote the
        # same file concurrently), optimizer state kept for resumption
        if epoch % hypes['train_params']['save_freq'] == 0 and opt.rank == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            torch.save({'epoch': epoch + 1, 'optimizer': optimizer.state_dict()},
                       opt_state_path)
            
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('[epoch %d/%d] validation loss %f' % (epoch + 1, epoches,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            # CoopFGNN patch: rank 0 alone decides and writes the best-val
            # checkpoint (each rank only sees its shard of the val set)
            if valid_ave_loss < lowest_val_loss and opt.rank == 0:
                lowest_val_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    if opt.rank == 0:
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1
                with open(os.path.join(saved_path, 'best_val.json'), 'w') as f:
                    json.dump({'loss': lowest_val_loss, 'epoch': lowest_val_epoch}, f)

        if not per_iter_lr:
            scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)
    # CoopFGNN patch: marker read by the auto-resubmit slurm script
    if opt.rank == 0:
        open(os.path.join(saved_path, 'TRAINING_DONE'), 'w').close()

    if opt.rank == 0:
        run_test = True
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
        if len(bestval_model_list) > 1:
            import numpy as np
            bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
            ascending_idx = np.argsort(bestval_model_epoch_list)
            for idx in ascending_idx:
                if idx != (len(bestval_model_list) - 1):
                    os.remove(bestval_model_list[idx])

        run_test=False
        if run_test:
            fusion_method = opt.fusion_method
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
            print(f"Running command: {cmd}")
            os.system(cmd)


if __name__ == '__main__':
    main()

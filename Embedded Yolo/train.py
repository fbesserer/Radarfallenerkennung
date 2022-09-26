import argparse
from typing import List, Any, Dict
import time

import torch
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import init_seeds, parse_data_cfg
from dataset import ImagesAndLabels, collate_fn
from network import EmbeddedYolo
from boxtargets import BoxTarget
from evaluate import evaluate


def accumulate_predictions(predictions):
    all_predictions = [predictions]

    # if get_rank() != 0:
    #     return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions


def format_pred(preds: List[BoxTarget]) -> List[Dict[str, Tensor]]:
    # format needed for torchmetrics evaluation
    predictions: list = []
    for pred in preds:
        d: dict = {}
        d['boxes'] = pred.box
        d['scores'] = pred.fields.get('scores')
        d['labels'] = pred.fields['labels']
        predictions.append(d)
    return predictions


@torch.no_grad()
def valid(loader, valid_set, model, device, val=True):
    torch.cuda.empty_cache()
    model.eval()
    pbar = tqdm(loader, dynamic_ncols=True)

    preds: list = []
    targets_all: list = []
    loss_mean = torch.zeros(4)

    # collect all predictions
    for batch_nr, (images, targets, ids) in enumerate(pbar):
        model.zero_grad()

        targets = [t.to('cuda') for t in targets]
        target: List[Dict[str, Tensor]] = format_pred(targets)
        targets_all.extend(target)

        images = images.to(device)
        pred: List[BoxTarget]
        pred, loss_dict = model(images.tensors, images.sizes, targets=targets)
        pred = [p.to('cuda') for p in pred]
        pred: List[Dict[str, Tensor]] = format_pred(pred)
        preds.extend(pred)

        # _, loss_dict = model(images.tensors, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        loss = loss_cls + loss_box + loss_center

        loss_mean[0] = (batch_nr * loss_mean[0] + loss) / (batch_nr + 1)
        loss_mean[1] = (batch_nr * loss_mean[1] + loss_cls) / (batch_nr + 1)
        loss_mean[2] = (batch_nr * loss_mean[2] + loss_box) / (batch_nr + 1)
        loss_mean[3] = (batch_nr * loss_mean[3] + loss_center) / (batch_nr + 1)

        pbar.set_description(
            (
                f'epoch: {epoch}; cls: {loss_cls:.4f}; '
                f'box: {loss_box:.4f}; center: {loss_center:.4f}'
            )
        )

    # evaluate all predictions with their respective targets
    start = time.perf_counter()
    metrics: Dict[str, Tensor] = evaluate(preds, targets_all)
    print(f"evaluation time: {round(time.perf_counter() - start, 2)} s")

    if tb_writer:
        # metrics
        prefix = "val/" if val else "train/"
        metrics: List = sorted(metrics.items(), key=lambda x: x[0])
        tags = [prefix + key[0] for key in metrics]
        tensors = [val[1] for val in metrics]
        for x, tag in zip(tensors, tags):
            tb_writer.add_scalar(tag, x, epoch)

        # losses
        tags = [prefix + 'loss', prefix + 'cls_loss', prefix + 'box_loss', prefix + 'center_loss']
        for x, tag in zip(list(loss_mean), tags):
            tb_writer.add_scalar(tag, x, epoch)
        tb_writer.flush()


def train(epoch, loader, model, optimizer, device):
    model.train()
    pbar = tqdm(loader, dynamic_ncols=True)

    loss_mean = torch.zeros(4)

    for batch_nr, (images, targets, _) in enumerate(pbar):
        model.zero_grad()  # alle Gradienten auf 0 sonst werden diese akkumuliert aus vorherigem batch

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        _, loss_dict = model(images.tensors, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()

        loss = loss_cls + loss_box + loss_center
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        loss_mean[0] = (batch_nr * loss_mean[0] + loss) / (batch_nr + 1)
        loss_mean[1] = (batch_nr * loss_mean[1] + loss_cls) / (batch_nr + 1)
        loss_mean[2] = (batch_nr * loss_mean[2] + loss_box) / (batch_nr + 1)
        loss_mean[3] = (batch_nr * loss_mean[3] + loss_center) / (batch_nr + 1)

        pbar.set_description(
            (
                f'epoch: {epoch}; cls: {loss_cls:.4f}; '
                f'box: {loss_box:.4f}; center: {loss_center:.4f}'
            )
        )
    if tb_writer:
        tags = ['train/loss', 'train/cls_loss', 'train/box_loss', 'train/center_loss']
        for x, tag in zip(list(loss_mean), tags):
            tb_writer.add_scalar(tag, x, epoch)
        tb_writer.flush()

    if epoch == opt.epochs - 1:
        valid(loader, None, model, device, val=False)  # evaluation (metrics) der trainingsdaten


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=64)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--data', type=str, default='data/radar.data', help='*.data path')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov4-tiny.weights', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    opt = parser.parse_args()
    opt.n_class = 5
    opt.threshold = 0.05
    opt.top_n = 1000
    opt.nms_threshold = 0.6
    opt.post_top_n = 100
    opt.min_size = 0

    device = 'cuda' if torch.cuda.is_available() == 1 else 'cpu'
    # device = 'cpu'

    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size

    tb_writer = SummaryWriter()

    init_seeds()  # eventuell raus lassen später?
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    valid_path = data_dict['valid']

    train_set = ImagesAndLabels(train_path, cache_images=opt.cache_images)
    valid_set = ImagesAndLabels(valid_path)
    batch_size = min(batch_size, len(train_set))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        # sampler=data_sampler(train_set, shuffle=True, distributed=args.distributed), # notwendig?
        num_workers=0,
        collate_fn=collate_fn(),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        # shuffle=True,
        # sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        num_workers=0,
        collate_fn=collate_fn(),
    )

    model = EmbeddedYolo(opt)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum=0.9,
        weight_decay=opt.l2,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16, 22], gamma=0.1
    )

    for epoch in range(opt.epochs):
        train(epoch, train_loader, model, optimizer, device)
        valid(valid_loader, valid_set, model, device)

        scheduler.step()
        # if get_rank() == 0:

        torch.save(
            {'model': model.state_dict(), 'optim': optimizer.state_dict()},
            f'checkpoint/epoch-{epoch + 1}.pt',
        )

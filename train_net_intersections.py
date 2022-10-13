#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
import json
import random
import collections
import lmdb
import argparse

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin

from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

# for LooseVersion issue, see https://github.com/pytorch/pytorch/pull/69904/commits/9af2edb158b3603c44eff6e12896f1d215e8b898


ckpt_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_r101-fpn-3x_converted.pth'))
cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'MSCOCO2017'))
intersections_basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Intersections'))
video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes_coco = [['person'], ['car', 'bus', 'truck']]
thing_classes = ['person', 'vehicle']
assert len(thing_classes_coco) == len(thing_classes)



def get_coco_dicts(split):
    if split == 'valid':
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_val2017.json')
    elif split == 'train':
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_train2017.json')
    else: return None
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    category_id_remap = {}
    for cat in annotations['categories']:
        for i in range(0, len(thing_classes_coco)):
            if cat['name'] in thing_classes_coco[i]:
                category_id_remap[cat['id']] = i
    coco_dicts = {}
    images_dir = os.path.join(cocodir, 'images', 'val2017' if split == 'valid' else 'train2017')
    for im in annotations['images']:
        coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
    for ann in annotations['annotations']:
        if not ann['category_id'] in category_id_remap:
            continue
        coco_dicts[ann['image_id']]['annotations'].append({'bbox': ann['bbox'], 'iscrowd': ann['iscrowd'], 'bbox_mode': BoxMode.XYWH_ABS, 'segmentation': [], 'category_id': category_id_remap[ann['category_id']]})
    coco_dicts = list(coco_dicts.values())
    coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
    for i in range(0, len(coco_dicts)):
        coco_dicts[i]['image_id'] = i + 1
    count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('MSCOCO-2017 %s: %d images, %d bboxes' % (split, count_images, count_bboxes))
    return coco_dicts


def get_unlabeled_dicts(args):
    lmdb_path = os.path.normpath(os.path.join(intersections_basedir, 'images', 'train_lmdb', args.id))
    with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
        meta = json.load(fp)
    ifilelist = meta['ifilelist']
    dict_json = []
    for i in range(0, len(ifilelist)):
        dict_json.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': []})
    print('unlabeled frames of video %s at %s: %d images' % (args.id, lmdb_path, len(dict_json)))
    return dict_json


def all_unlabeled_dicts(args, total_images):
    random.seed(42)
    images_per_video_cap = int(total_images / len(video_id_list))
    dict_json_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        dict_json_v = get_unlabeled_dicts(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])
        dict_json_all = dict_json_all + dict_json_v
    args.id = id_back
    for i in range(0, len(dict_json_all)):
        dict_json_all[i]['image_id'] = i + 1
    print('all videos %d images' % len(dict_json_all))
    return dict_json_all


def get_manual_dicts(args):
    inputdir = os.path.join(intersections_basedir, 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    return annotations


def all_annotation_dicts(args):
    annotations_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        annotations_all = annotations_all + get_manual_dicts(args)
    args.id = id_back
    for i in range(0, len(annotations_all)):
        annotations_all[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(annotations_all), sum(list(map(lambda x: len(x['annotations']), annotations_all)))))
    return annotations_all


def main(args):
    config_yaml = os.path.normpath(os.path.join(os.path.dirname(__file__), 'configs', 'faster_rcnn_R101_FPN_cross_intersections.yaml'))
    print('load configuration from:', config_yaml)

    DatasetCatalog.register('mscoco2017_train_remap', lambda: get_coco_dicts('train'))
    DatasetCatalog.register('mscoco2017_valid_remap', lambda: get_coco_dicts('valid'))
    MetadataCatalog.get('mscoco2017_train_remap').thing_classes = thing_classes
    MetadataCatalog.get('mscoco2017_valid_remap').thing_classes = thing_classes
    MetadataCatalog.get('mscoco2017_valid_remap').evaluator_type = 'coco'

    dst_unlabel, dst_manual = 'intersection_unlabeled_%s' % args.id, 'intersection_manual_%s' % args.id
    if args.id in video_id_list:
        DatasetCatalog.register(dst_unlabel, lambda: get_unlabeled_dicts(args))
        DatasetCatalog.register(dst_manual, lambda: get_manual_dicts(args))
    elif args.id == 'compound':
        DatasetCatalog.register(dst_unlabel, lambda: all_unlabeled_dicts(args, args.batch_size * args.iters))
        DatasetCatalog.register(dst_manual, lambda: all_annotation_dicts(args))
    else:
        raise NotImplementedError
    MetadataCatalog.get(dst_unlabel).thing_classes = thing_classes
    MetadataCatalog.get(dst_manual).thing_classes = thing_classes
    MetadataCatalog.get(dst_manual).evaluator_type = 'coco'

    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(config_yaml)
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.DATASETS.TRAIN_LABEL = ('mscoco2017_train_remap',)
    cfg.DATASETS.TRAIN_UNLABEL = (dst_unlabel,)
    # cfg.DATASETS.TEST = (dst_manual,) # debug
    # cfg.DATASETS.TEST = ('mscoco2017_valid_remap', dst_manual)

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.IMG_PER_BATCH_LABEL = args.batch_size
    cfg.SOLVER.IMG_PER_BATCH_UNLABEL = args.batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.freeze()
    print('load weights from:', cfg.MODEL.WEIGHTS)

    Trainer = ATeacherTrainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), 'adapt_intersections_%s_lr%.5f_iter%d.pth' % (args.id, cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_ITER)))


def convert_ckpt(args):
    sd = torch.load(os.path.join(os.path.dirname(__file__), '..', 'Intersections', 'models', 'mscoco2017_remap_r101-fpn-3x.pth'))
    sd_converted = type(sd)()
    for k in sd:
        sd_converted['modelTeacher.' + k] = sd[k]
    for k in sd:
        sd_converted['modelStudent.' + k] = sd[k]
    torch.save(sd_converted, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptation Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--eval_interval', type=int, default=4010)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    parser.add_argument('--ddp_port', type=int, default=50152)
    args = parser.parse_args()

    if args.opt == 'adapt':
        if args.ddp_num_gpus > 1:
            launch(main, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='tcp://127.0.0.1:%d' % args.ddp_port, args=(args,))
        else:
            main(args)
    elif args.opt == 'convert':
        convert_ckpt(args)

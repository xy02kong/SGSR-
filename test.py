import os
import time
import torch
import numpy as np

# 兼容补丁：NumPy>=1.24 已删除 np.int/np.float/np.bool 等旧别名，
# 而 spikingjelly 0.0.0.0.14 内部仍在使用，这里补回以避免 AttributeError。
for _np_alias, _py_type in (("int", int), ("float", float), ("bool", bool),
                            ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _np_alias):
        setattr(np, _np_alias, _py_type)

import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou, scores_from_confmat
from utils.convert_state import convert_state_dict

from dataset.event.base_trainer import BaseTrainer
from spikingjelly.activation_based import layer, neuron, functional
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image
from torchvision.utils import save_image
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth,\
    ProbOhemCrossEntropy2d, FocalLoss2d

# # 检查是否有可用的 GPU
# if torch.cuda.is_available():
#     device = torch.device("cuda:3")
# else:
#     device = torch.device("cpu")

def parse_args():
    parser = ArgumentParser(description='Efficient Event-based Semantic Segmentation with Spike-driven Lightweight Transformer-based Networks')
    parser.add_argument('--model', default="SGSR", help="model name: (default SGSR)")
    parser.add_argument('--dataset', default="DDD17_events", help="dataset: DDD17_events or DSEC_events")
    parser.add_argument('--input_size', type=str, default="200,346", help="DDD17_events:200,346,DSEC_events:480,640")
    parser.add_argument('--dataset_path', type=str, default="/home/zhuxx/datasets/DDD17_events")
    parser.add_argument('--workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=2,
                        help=" the batch_size is set to 64 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="pretrained_models/DDD17/STLNet_DDD17_Test.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--save', type=bool, default=True, help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="3", type=str, help="gpu ids (default: 0)")

    parser.add_argument('--split', type=str, default="train", help="spilt in ['train', 'test', 'valid']")
    parser.add_argument('--nr_events_data', type=int, default=1)
    parser.add_argument('--delta_t_per_data', type=int, default=50)
    parser.add_argument('--nr_events_window', type=int, default=100000, help='DDD17:32000,DSEC:100000')
    parser.add_argument('--data_augmentation_train', type=bool, default=True)
    parser.add_argument('--event_representation', type=str, default="voxel_grid")
    parser.add_argument('--nr_temporal_bins', type=int, default=5)
    parser.add_argument('--require_paired_data_train', type=bool, default=False)
    parser.add_argument('--require_paired_data_val', type=bool, default=False)
    parser.add_argument('--separate_pol', type=bool, default=False)
    parser.add_argument('--normalize_event', type=bool, default=True)
    parser.add_argument('--fixed_duration', type=bool, default=True)

    # event datasets
    parser.add_argument('--use_ohem', type=bool, default=True, help='OhemCrossEntropy2d Loss for event dataset')
    parser.add_argument("--use_earlyloss", type=bool, default=True, help='Use early-surpervised training for event dataset')
    parser.add_argument("--balance_weights", type=list, default=[1.0, 0.4], help='balance between out and early_out')

    args = parser.parse_args()

    return args




# def test(args, test_loader, model, save=True):
#     """
#     args:
#       test_loader: loaded for test dataset
#       model: model
#     return: class IoU and mean IoU
#     """
#     # evaluation or test mode
#     model.eval()                          
#     total_batches = len(test_loader)

#     data_list = []
#     for i, (input, image, label, _) in enumerate(test_loader):
#         with torch.no_grad():
#             input_var = input.cuda()
            
#         start_time = time.time()
#         output = model(input_var)
#         torch.cuda.synchronize()
#         time_taken = time.time() - start_time
#         print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))

#         # save the predicted image
#         if save:
#             save_predict(output, label, image, i, args.dataset, args.save_seg_dir,
#                          output_grey=False, output_color=True, gt_color=True)

#         output = output.cpu().data[0].numpy()
#         gt = np.asarray(label[0].numpy(), dtype=np.uint8)
#         output = output.transpose(1, 2, 0)
#         output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
#         data_list.append([gt.flatten(), output.flatten()])

#     meanIoU, per_class_iu, acc = get_iou(data_list, args.classes)
#     return meanIoU, per_class_iu



def test(args, val_loader, model, criterion, device):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    epoch_loss = []
    nclass = args.classes
    # 混淆矩阵常驻 GPU 累加，避免每张图 GPU→CPU 搬运与同步。
    conf = torch.zeros(nclass, nclass, dtype=torch.int64, device=device)
    for i, (event, label, _) in enumerate(val_loader):
        start_time = time.time()
        with torch.no_grad():
            # input_var = Variable(input).cuda(device)
            event_var = event.cuda(device)
            label_var = label.long().cuda(device)
            outputs = model(event_var)

            if not isinstance(outputs, dict):
                raise TypeError(
                    f"SGSR must return dict, got {type(outputs)}"
                )

            seg_logits = outputs["seg"]
            # loss:[loss, ohem_loss, early_loss]
            loss = criterion(seg_logits, label_var)
        functional.reset_net(model)
        time_taken = time.time() - start_time
        print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
        epoch_loss.append(loss[0].item())

        pred = seg_logits.argmax(dim=1)  # [B,H,W] on GPU

        # 直接在 GPU 上统计整个 batch 的混淆矩阵并累加（原实现只取 batch 第一张）。
        valid = (label_var >= 0) & (label_var < nclass)
        idx = label_var[valid] * nclass + pred[valid]
        conf += torch.bincount(idx, minlength=nclass * nclass).reshape(nclass, nclass)

    conf = conf.cpu().numpy()
    meanIoU, per_class_iu, acc = scores_from_confmat(conf, nclass)
    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    return meanIoU, per_class_iu, acc, average_epoch_loss_val



def test_no_val(args, test_loader, model, save_path, device, save=True):
    if args.dataset == 'DDD17_events':
        # DDD17:['flat','background','object','vegetation','human','vehicle']
        color_map = [(128, 64,128), #紫
                    (70 , 70, 70), #灰
                    (220,220,  0), #黄
                    (107,142, 35), #绿
                    (220, 20, 60), #红
                    (0  ,  0,142)] #蓝
    elif args.dataset == 'DSEC_events':
        # DSEC:['background','building','fence','person','pole','road','sidewalk','vegetation','car','wall','traffic sign']
        color_map = [(0,  0,  0),
                    (70 ,70, 70),
                    (190,153,153),
                    (220, 20,60),
                    (153,153,153),
                    (128, 64,128),
                    (244, 35,232),
                    (107,142, 35),
                    (0,  0,  142),
                    (102,102,156),
                    (220,220,  0)]
    
    for index, batch in enumerate(tqdm(test_loader)):
        event, image, label, _, _ = batch
        size = label.size()
        event = event.cuda(device)

        outputs = model(event)

        if not isinstance(outputs, dict):
            raise TypeError(
                f"SGSR must return dict, got {type(outputs)}"
            )

        seg_logits = outputs["seg"]

        if seg_logits.size()[-2] != size[0] or seg_logits.size()[-1] != size[1]:
            seg_logits = F.interpolate(
                seg_logits, size[-2:],
                mode='bilinear', align_corners=True
            )
        pred = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        label = label.squeeze(0).cpu().numpy()
        
        if save:
            sv_predict = np.zeros((size[-2], size[-1], 3), dtype=np.uint8)
            sv_label = np.zeros((size[-2], size[-1], 3), dtype=np.uint8)
            
            sv_path_pred = os.path.join(save_path,'pred')
            sv_path_label = os.path.join(save_path,'label')
            sv_path_image = os.path.join(save_path,'image')
            
            if not os.path.exists(sv_path_pred):
                os.mkdir(sv_path_pred)
            if not os.path.exists(sv_path_label):
                os.mkdir(sv_path_label)
            if not os.path.exists(sv_path_image):
                os.mkdir(sv_path_image)
            
            # 验证全部valid数据集
            if len(pred.shape) != 3:
                pred = pred[np.newaxis, :, :]
                label = label[np.newaxis, :, :]
            
            for idx in range(pred.shape[0]):
                for i, color in enumerate(color_map):
                    for j in range(3):
                        sv_predict[:,:,j][pred[idx]==i] = color_map[i][j]
                        sv_label[:,:,j][label[idx]==i] = color_map[i][j]
                
                sv_predict_event = Image.fromarray(sv_predict)
                sv_label_event = Image.fromarray(sv_label)
                sv_predict_event.save(sv_path_pred+"/predict{}.png".format(index * pred.shape[0] + idx))
                sv_label_event.save(sv_path_label+"/label{}.png".format(index * pred.shape[0] + idx))
                save_image(image[idx], sv_path_image+'/image{}.png'.format(index * pred.shape[0] + idx))



def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    h, w = map(int, args.input_size.split(','))
    print(args)

    # 设置目标设备
    if torch.cuda.is_available():
        gpu_ids = [int(id) for id in args.gpus.split(',')]
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        if args.cuda:
            print("=====> use gpu id: '{}'".format(args.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            if not torch.cuda.is_available():
                raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    in_channels = 2 if args.separate_pol else 1
    model = build_model(
        args.model,
        num_classes=args.classes,
        in_channels=in_channels,
        temporal_steps=args.nr_temporal_bins,
    )
    step_modules = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "step_mode")
    ]
    invalid_step_modules = [
        (name, module.__class__.__name__, module.step_mode)
        for name, module in step_modules
        if module.step_mode != "m"
    ]
    if invalid_step_modules:
        details = "\n".join(
            f"  {name}: {cls_name}(step_mode={step_mode!r})"
            for name, cls_name, step_mode in invalid_step_modules
        )
        raise RuntimeError(
            "Expected an intrinsically multi-step model, but found "
            f"non-'m' modules:\n{details}"
        )
    print(f"=====> verified multi-step modules: {len(step_modules)}")

    
    if args.dataset == 'camvid':
        if args.use_label_smoothing:
            criteria = CrossEntropyLoss2dLabelSmooth(weight=None, ignore_label=ignore_label)
        else:
            criteria = CrossEntropyLoss2d(weight=None, ignore_label=ignore_label)
        
    elif args.dataset == 'cityscapes':
        if args.use_ohem:
            min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
            criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
        elif args.use_label_smoothing:
            criteria = CrossEntropyLoss2dLabelSmooth(weight=None, ignore_label=ignore_label)
        elif args.use_lovaszsoftmax:
            criteria = LovaszSoftmax(ignore_index=ignore_label)
        elif args.use_focal:
            criteria = FocalLoss2d(weight=None, ignore_index=ignore_label)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)
    
    # 不知道event数据集各类的weights, use_weight=False
    elif args.dataset in ['DDD17_events', 'DSEC_events']:
        if args.use_ohem:
            min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
            criteria = ProbOhemCrossEntropy2d(use_weight=False, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept, balance_weights=args.balance_weights)
        elif args.use_focal:
            criteria = FocalLoss2d(weight=None, ignore_index=ignore_label, balance_weights=args.balance_weights)
        else:
            criteria = CrossEntropyLoss2d(weight=None, ignore_label=ignore_label)

    
    
    if args.cuda:
        model = model.cuda(device)  # using GPU for inference
        criteria = criteria.cuda(device)
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)


    # DDD17/DSEC datasets
    base_trainer_instance = BaseTrainer()
    trainLoader, testLoader = base_trainer_instance.createDataLoaders(args)

    
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning validation")
    print("validation set length: ", len(testLoader))
    # test_no_val(args, testLoader, model, args.save_seg_dir, save=False)
    # test(args, testLoader, model, save=False)
    
    # 需要计算miou值
    mIOU_val, per_class_iu, _, _ = test(args, testLoader, model, criteria, device)
    print("mIOU_val:",mIOU_val)
    print("per_class_iu:",per_class_iu)

    # # Save the result
    # args.logFile = 'test.txt'
    # logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # # Save the result
    # if os.path.isfile(logFileLoc):
    #     logger = open(logFileLoc, 'a')
    # else:
    #     logger = open(logFileLoc, 'w')
    #     logger.write("Mean IoU: %.4f" % mIOU_val)
    #     logger.write("\nPer class IoU: ")
    #     for i in range(len(per_class_iu)):
    #         logger.write("%.4f\t" % per_class_iu[i])
    # logger.flush()
    # logger.close()


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'DDD17_events':
        args.classes = 6
    elif args.dataset == 'DSEC_events':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)
    
    ignore_label = 255
    test_model(args)

# -*- coding: utf-8 -*-
"""
PyTorch vs MindSpore GoogLeNet 对比实验：
1) pth 转 ckpt
2) ImageNet 输入（默认 500 张），主 logits 误差 + 预测一致率

用法: python googlenet/compare_torch_ms.py [--num-samples N] [--skip-convert] [--imagenet-dir DIR]
"""
import os
import sys
import argparse
import numpy as np
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PTH = os.path.join(ROOT, "googlenet", "googlenet-1378be20.pth")
CKPT = os.path.join(ROOT, "googlenet", "googlenet.ckpt")
IMAGENET_ROOT = os.path.join(ROOT, "imagenet")
# ILSVRC2012 验证集标签：第 i 行(0-based)=图像 0000000i.JPEG，值为 ILSVRC2012_ID (1~1000)
GROUND_TRUTH_FILE = os.path.join(ROOT, "imagenet", "ILSVRC2012_validation_ground_truth.txt")
# meta.mat：ILSVRC2012_ID -> torchvision index (0~999)，二者顺序不同
META_MAT = os.path.join(ROOT, "imagenet", "meta.mat")
IMAGE_SIZE = 224


def convert_pth_to_ckpt():
    from googlenet.pth_to_ckpt import convert
    convert()


def _find_imagenet_dir():
    for sub in ("ILSVRC2012_img_val", "ILSVRC2013_DET_val", "val", "val2012", ""):
        d = os.path.join(IMAGENET_ROOT, sub) if sub else IMAGENET_ROOT
        if not os.path.isdir(d):
            continue
        jpegs = glob.glob(os.path.join(d, "*.JPEG")) or glob.glob(os.path.join(d, "*.jpg"))
        if jpegs:
            return d
    return None


def _path_to_val_index(p):
    """从文件名解析 val 序号，如 ILSVRC2012_val_00000001.JPEG -> 1。"""
    base = os.path.basename(p)
    for s in base.replace(".JPEG", "").replace(".jpg", "").split("_"):
        if s.isdigit():
            return int(s)
    return 0


def _imagenet_paths_sorted_by_index(imagenet_dir, num_samples):
    """图片路径按 val 序号排序，取前 num_samples 张（可能是非连续序号，如 1,7,18,...）。"""
    exts = ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(imagenet_dir, e)))
    paths = sorted(paths, key=_path_to_val_index)[:num_samples]
    return paths


def _load_ilsvrc_to_torchvision_map(meta_path):
    """从 meta.mat 构建 ILSVRC2012_ID (1~1000) -> torchvision index (0~999)。"""
    if not os.path.isfile(meta_path):
        return None
    try:
        import scipy.io
        mat = scipy.io.loadmat(meta_path)
        synsets = mat["synsets"]
        ilsvrc_to_wnid = {}
        for i in range(len(synsets)):
            row = synsets[i, 0]
            ilsvrc_id = int(row[0][0, 0])
            if 1 <= ilsvrc_id <= 1000:
                ilsvrc_to_wnid[ilsvrc_id] = str(row[1][0])
        wnids_sorted = sorted(ilsvrc_to_wnid.values())
        wnid_to_tv_idx = {wnid: i for i, wnid in enumerate(wnids_sorted)}
        return {ilsvrc_id: wnid_to_tv_idx[wnid] for ilsvrc_id, wnid in ilsvrc_to_wnid.items()}
    except Exception:
        return None


def get_labels_for_paths(gt_file, paths, meta_mat_path=None):
    """标签第 i 行(0-based)=图像 0000000(i+1).JPEG，故 line_idx = val_index - 1。用 meta.mat 将 ILSVRC_ID 转为 torchvision index。"""
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()[:50000]]
    gt_ilsvrc = np.array([int(x) for x in lines if x], dtype=np.int64)
    indices = np.array([_path_to_val_index(p) for p in paths], dtype=np.int64)
    line_indices = indices - 1  # val 1->line 0
    if np.any((line_indices < 0) | (line_indices >= len(gt_ilsvrc))):
        return None
    ilsvrc_ids = gt_ilsvrc[line_indices]
    meta_path = meta_mat_path or META_MAT
    ilsvrc_to_tv = _load_ilsvrc_to_torchvision_map(meta_path)
    if ilsvrc_to_tv is not None:
        return np.array([ilsvrc_to_tv.get(int(x), x - 1) for x in ilsvrc_ids], dtype=np.int64)
    return ilsvrc_ids - 1  # 回退：简单减 1


def get_imagenet_samples(num_samples, imagenet_dir=None):
    """ImageNet 图片目录，按 val 序号取前 num_samples 张，Resize 224 + CenterCrop + ToTensor [0,1]。返回 (samples, paths)。"""
    from PIL import Image
    import torchvision.transforms as T
    if imagenet_dir is None:
        imagenet_dir = _find_imagenet_dir()
    if not imagenet_dir or not os.path.isdir(imagenet_dir):
        raise FileNotFoundError("未找到 ImageNet 目录，请指定 --imagenet-dir 或放置于 %s" % IMAGENET_ROOT)
    paths = _imagenet_paths_sorted_by_index(imagenet_dir, num_samples)
    if len(paths) < num_samples:
        print("警告: 仅找到 %d 张图，请求 %d 张" % (len(paths), num_samples))
    tf = T.Compose([T.Resize(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE), T.ToTensor()])
    samples = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tf(img).numpy().astype(np.float32)
        samples.append(x[None])
    return np.concatenate(samples, axis=0), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=500, help="ImageNet 样本数（默认 500）")
    parser.add_argument("--skip-convert", action="store_true", help="跳过 pth->ckpt 转换")
    parser.add_argument("--imagenet-dir", type=str, default=None, help="ImageNet 图片目录")
    parser.add_argument("--ground-truth", type=str, default=GROUND_TRUTH_FILE, help="ILSVRC2012 验证集标签文件")
    parser.add_argument("--meta-mat", type=str, default=META_MAT, help="meta.mat 路径（ILSVRC->torchvision 映射，用于准确度）")
    args = parser.parse_args()

    if not args.skip_convert and os.path.isfile(PTH):
        print("========== 1. pth -> ckpt 转换 ==========")
        convert_pth_to_ckpt()
        print()

    import torch
    import mindspore as ms
    from mindspore import Tensor, context
    from torchvision.models import googlenet, GoogLeNet_Weights
    from googlenet.ms_googlenet import GoogLeNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).to(device).eval()
    if os.path.isfile(PTH):
        raw = torch.load(PTH, map_location=device, weights_only=True)
        pt_net.load_state_dict(raw, strict=False)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms_net = GoogLeNet(num_classes=1000)
    ms.load_checkpoint(CKPT, ms_net)
    ms_net.set_train(False)

    x_np, paths = get_imagenet_samples(args.num_samples, args.imagenet_dir)
    n = len(x_np)
    print("========== 2. 主 logits 误差（ImageNet） ==========")
    print("样本数: %d" % n)

    # 按每张图 val 序号取 GT，用 meta.mat 将 ILSVRC_ID 转为 torchvision index
    if os.path.isfile(args.ground_truth):
        gt = get_labels_for_paths(args.ground_truth, paths, args.meta_mat)
        if gt is not None:
            use_meta = "meta.mat 映射" if os.path.isfile(args.meta_mat) else "简单-1"
            print("使用标签数: %d（%s）" % (len(gt), use_meta))
        else:
            gt = None
            print("标签与图像序号不对齐，未计算准确度")
    else:
        gt = None
        print("未找到标签文件，不计算准确度: %s" % args.ground_truth)

    pt_logits_all, ms_logits_all = [], []
    batch_size = 32
    for i in range(0, len(x_np), batch_size):
        batch = x_np[i : i + batch_size]
        x_pt = torch.from_numpy(batch).to(device)
        x_ms = Tensor(batch, dtype=ms.float32)
        with torch.no_grad():
            pt_out = pt_net(x_pt)
            if isinstance(pt_out, tuple):
                pt_out = pt_out[0]
            pt_out = pt_out.cpu().numpy()
        ms_out = ms_net(x_ms).asnumpy()
        pt_logits_all.append(pt_out)
        ms_logits_all.append(ms_out)

    pt_logits = np.concatenate(pt_logits_all, axis=0)
    ms_logits = np.concatenate(ms_logits_all, axis=0)
    diff = np.abs(pt_logits - ms_logits)
    max_err = diff.max()
    mean_err = diff.mean()
    print("主 logits 最大绝对误差: %.6e" % max_err)
    print("主 logits 平均绝对误差: %.6e" % mean_err)

    pt_pred = pt_logits.argmax(axis=1)
    ms_pred = ms_logits.argmax(axis=1)
    match = (pt_pred == ms_pred).sum()
    print("预测一致数 / 总数: %d / %d (%.1f%%)" % (match, n, 100.0 * match / n))

    pt_correct = (pt_pred == gt).sum() if gt is not None and len(gt) == n else None
    ms_correct = (ms_pred == gt).sum() if gt is not None and len(gt) == n else None

    print("\n========== 准确度（Top-1，与 ground truth 对齐） ==========")
    if gt is not None and len(gt) == n:
        print("  PyTorch  准确度: %d / %d = %.2f%%" % (pt_correct, n, 100.0 * pt_correct / n))
        print("  MindSpore 准确度: %d / %d = %.2f%%" % (ms_correct, n, 100.0 * ms_correct / n))
    else:
        print("  未提供标签文件，未计算准确度。")

    print("\n========== 结果汇总 ==========")
    print("  主 logits 最大误差: %.6e" % max_err)
    print("  主 logits 平均误差: %.6e" % mean_err)
    print("  预测一致率: %.1f%%" % (100.0 * match / n))
    if pt_correct is not None and ms_correct is not None:
        print("  PyTorch  准确度: %.2f%%" % (100.0 * pt_correct / n))
        print("  MindSpore 准确度: %.2f%%" % (100.0 * ms_correct / n))


if __name__ == "__main__":
    main()

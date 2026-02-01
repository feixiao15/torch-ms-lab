# -*- coding: utf-8 -*-
"""
单张/多张图预测调试：打印 ground truth 与 PyTorch/MindSpore 预测，便于核对标签。
使用 meta.mat 将 ILSVRC2012_ID 转为 torchvision 类别索引（二者顺序不同）。

用法:
  python googlenet/predict_one_image.py           # 默认预测前 5 张
  python googlenet/predict_one_image.py -n 1      # 预测 1 张
  python googlenet/predict_one_image.py --image path/to/xxx.JPEG
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PTH = os.path.join(ROOT, "googlenet", "googlenet-1378be20.pth")
CKPT = os.path.join(ROOT, "googlenet", "googlenet.ckpt")
IMAGENET_ROOT = os.path.join(ROOT, "imagenet")
GROUND_TRUTH_FILE = os.path.join(ROOT, "imagenet", "ILSVRC2012_validation_ground_truth.txt")
META_MAT = os.path.join(ROOT, "imagenet", "meta.mat")
IMAGE_SIZE = 224


def _find_images(num_images):
    """按 val 序号排序，取前 num_images 张。"""
    import glob
    for sub in ("ILSVRC2012_img_val", "ILSVRC2013_DET_val", "val", "val2012", ""):
        d = os.path.join(IMAGENET_ROOT, sub) if sub else IMAGENET_ROOT
        if not os.path.isdir(d):
            continue
        jpegs = glob.glob(os.path.join(d, "*.JPEG")) or glob.glob(os.path.join(d, "*.jpg"))
        if jpegs:
            jpegs = sorted(jpegs, key=lambda p: _val_index_from_path(p))[:num_images]
            return jpegs
    return []


def _val_index_from_path(p):
    base = os.path.basename(p)
    for s in base.replace(".JPEG", "").replace(".jpg", "").split("_"):
        if s.isdigit():
            return int(s)
    return 0


def _load_ilsvrc_to_torchvision_map(meta_path):
    """
    从 meta.mat 构建 ILSVRC2012_ID (1~1000) -> torchvision index (0~999)。
    meta.mat 有 synsets，含 ILSVRC2012_ID 与 WNID。torchvision 按 WNID 字母序排列。
    """
    if not os.path.isfile(meta_path):
        return None
    try:
        import scipy.io
        mat = scipy.io.loadmat(meta_path)
        synsets = mat["synsets"]
        # synsets 可能含 1860 条，只取 ILSVRC2012_ID 1~1000 的 1000 个竞赛类
        ilsvrc_to_wnid = {}
        for i in range(len(synsets)):
            row = synsets[i, 0]
            ilsvrc_id = int(row[0][0, 0])
            if 1 <= ilsvrc_id <= 1000:
                ilsvrc_to_wnid[ilsvrc_id] = str(row[1][0])
        # torchvision 顺序：按 wnid 字母序
        wnids_sorted = sorted(ilsvrc_to_wnid.values())
        wnid_to_tv_idx = {wnid: i for i, wnid in enumerate(wnids_sorted)}
        return {ilsvrc_id: wnid_to_tv_idx[wnid] for ilsvrc_id, wnid in ilsvrc_to_wnid.items()}
    except Exception as e:
        print("警告: 无法加载 meta.mat (%s)，将用 ILSVRC_ID-1 近似" % e)
        return None


def _get_gt_ilsvrc_id(gt_file, val_index):
    """标签第 i 行(0-based)=图像 0000000(i+1).JPEG。val_index 1->line 0, val_index 2->line 1。"""
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    line_idx = val_index - 1  # val 1->line 0
    if line_idx < 0 or line_idx >= len(lines):
        return None
    return int(lines[line_idx].strip())


def _get_class_name(idx_0_999):
    """torchvision ImageNet 0-999 对应的类别名。"""
    try:
        from torchvision.models import GoogLeNet_Weights
        meta = getattr(GoogLeNet_Weights.IMAGENET1K_V1, "meta", None)
        if meta and "categories" in meta:
            c = meta["categories"][idx_0_999]
            return c[1] if isinstance(c, (list, tuple)) and len(c) > 1 else str(c)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="多张图预测，打印 GT 与 PT/MS 预测（使用 meta.mat 校正 ILSVRC->torchvision 映射）")
    parser.add_argument("-n", "--num-images", type=int, default=5, help="预测前 N 张图（默认 5）")
    parser.add_argument("--image", type=str, default=None, help="指定单张图片路径（覆盖 -n）")
    parser.add_argument("--ground-truth", type=str, default=GROUND_TRUTH_FILE, help="标签文件")
    parser.add_argument("--meta-mat", type=str, default=META_MAT, help="meta.mat 路径（ILSVRC->torchvision 映射）")
    args = parser.parse_args()

    # 加载 ILSVRC2012_ID -> torchvision index 映射
    ilsvrc_to_tv = _load_ilsvrc_to_torchvision_map(args.meta_mat)
    if ilsvrc_to_tv is None:
        ilsvrc_to_tv = {i: i - 1 for i in range(1, 1001)}  # 回退：简单减 1

    # 确定要预测的图片
    if args.image and os.path.isfile(args.image):
        image_paths = [os.path.abspath(args.image)]
    else:
        image_paths = _find_images(args.num_images)
        if not image_paths:
            print("未找到任何 ImageNet 图片，请指定 --image 或把图片放到 imagenet/ 下")
            return
        print("预测前 %d 张图（按 val 序号排序）\n" % len(image_paths))

    from PIL import Image
    import torchvision.transforms as T
    import numpy as np

    tf = T.Compose([T.Resize(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE), T.ToTensor()])
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    import torch
    from torchvision.models import googlenet, GoogLeNet_Weights
    pt_net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).to(device).eval()
    if os.path.isfile(PTH):
        raw = torch.load(PTH, map_location=device, weights_only=True)
        pt_net.load_state_dict(raw, strict=False)

    import mindspore as ms
    from mindspore import Tensor, context
    from googlenet.ms_googlenet import GoogLeNet
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms_net = GoogLeNet(num_classes=1000)
    ms.load_checkpoint(CKPT, ms_net)
    ms_net.set_train(False)

    pt_correct, ms_correct, total = 0, 0, 0
    for idx, image_path in enumerate(image_paths):
        val_index = _val_index_from_path(image_path)
        print("=" * 60)
        print("[%d] %s" % (idx + 1, os.path.basename(image_path)))
        print("    val 序号: %d" % val_index)

        # Ground truth
        gt_ilsvrc = _get_gt_ilsvrc_id(args.ground_truth, val_index) if os.path.isfile(args.ground_truth) else None
        if gt_ilsvrc is not None:
            gt_tv = ilsvrc_to_tv.get(gt_ilsvrc, gt_ilsvrc - 1)
            gt_name = _get_class_name(gt_tv)
            print("    GT: ILSVRC_ID=%d -> torchvision_idx=%d  %s" % (gt_ilsvrc, gt_tv, gt_name or ""))
        else:
            gt_tv = None
            print("    GT: 无效（行号超出）")

        # 预处理
        img = Image.open(image_path).convert("RGB")
        x_np = tf(img).numpy().astype(np.float32)[None]

        # PyTorch
        with torch.no_grad():
            pt_out = pt_net(torch.from_numpy(x_np).to(device))
            if isinstance(pt_out, tuple):
                pt_out = pt_out[0]
            pt_pred = int(pt_out.cpu().numpy()[0].argmax())
        pt_name = _get_class_name(pt_pred)
        print("    PT: idx=%d  %s" % (pt_pred, pt_name or ""))

        # MindSpore
        ms_out = ms_net(Tensor(x_np, dtype=ms.float32)).asnumpy()[0]
        ms_pred = int(ms_out.argmax())
        ms_name = _get_class_name(ms_pred)
        print("    MS: idx=%d  %s" % (ms_pred, ms_name or ""))

        if gt_tv is not None:
            total += 1
            if pt_pred == gt_tv:
                pt_correct += 1
            if ms_pred == gt_tv:
                ms_correct += 1
            print("    %s PT%s MS%s" % ("✓" if pt_pred == gt_tv and ms_pred == gt_tv else "✗",
                                       "✓" if pt_pred == gt_tv else "✗", "✓" if ms_pred == gt_tv else "✗"))
        print()

    if total > 0:
        print("========== 汇总（%d 张） ==========" % total)
        print("  PT 正确: %d/%d = %.1f%%" % (pt_correct, total, 100.0 * pt_correct / total))
        print("  MS 正确: %d/%d = %.1f%%" % (ms_correct, total, 100.0 * ms_correct / total))


if __name__ == "__main__":
    main()

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse
import random
from sam2.build_sam import build_sam2_video_predictor
from torchvision.transforms import ToPILImage
from medpy.metric import binary
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

WEIGHT_FILES = [
    "../sscd-copy-detection/sscd/models_weight/sscd_disc_advanced.torchscript.pt",
    "../sscd-copy-detection/sscd/models_weight/sscd_disc_blur.torchscript.pt",
    "../sscd-copy-detection/sscd/models_weight/sscd_disc_large.torchscript.pt",
    "../sscd-copy-detection/sscd/models_weight/sscd_disc_mixup.torchscript.pt",
    "../sscd-copy-detection/sscd/models_weight/sscd_imagenet_advanced.torchscript.pt",
    "../sscd-copy-detection/sscd/models_weight/sscd_imagenet_blur.torchscript.pt",
    "../sscd-copy-detection/sscd/models_weight/sscd_imagenet_mixup.torchscript.pt",
]

class MaskedImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = self._load_image_files()
        
    def _load_image_files(self):
        benign_path = os.path.join(self.image_dir, 'benign')
        malignant_path = os.path.join(self.image_dir, 'malignant')
        
        benign_files = [os.path.join(benign_path, f) 
                       for f in os.listdir(benign_path) 
                       if '_mask' not in f]
                       
        malignant_files = [os.path.join(malignant_path, f) 
                          for f in os.listdir(malignant_path) 
                          if '_mask' not in f]
                          
        all_files = benign_files + malignant_files
        all_files.sort()
        return all_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = image_path.replace('.png', '_mask.png')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        np_image = np.array(image)
        np_mask = np.array(mask)
        np_mask = np.where(np_mask > 128, 255, 0)

        if self.transform:
            image = self.transform(image)

        return image, np_image, np_mask, image_path

def load_datasets(train_dir, val_dir, test_dir, batch_sizes=(1, 1, 1)):
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets = {
        'train': MaskedImageDataset(train_dir, transform),
        'val': MaskedImageDataset(val_dir, transform),
        'test': MaskedImageDataset(test_dir, transform)
    }
    
    print(f"Train Dataset: {len(datasets['train'])}")
    print(f"Validation Dataset: {len(datasets['val'])}")
    print(f"Test Dataset: {len(datasets['test'])}")

    loaders = [
        DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        for dataset, bs in zip(datasets.values(), batch_sizes)
    ]
    
    return loaders

def load_model(model_path, device):
    model = torch.jit.load(model_path)
    model.to(device).eval()
    return model

@torch.no_grad()
def extract_embeddings(model, data_loader, device):
    embeddings, images_list, masks_list, paths_list = [], [], [], []
    
    for batch in data_loader:
        images, np_images, np_masks, image_paths = batch
        embeddings.append(model(images.to(device)))
        images_list.append(np_images)
        masks_list.append(np_masks)
        paths_list.extend(image_paths)
    return torch.cat(embeddings, dim=0), torch.cat(images_list, dim=0), torch.cat(masks_list, dim=0), paths_list

def get_top_k_similarities(val_embedding, train_embeddings, train_images, train_masks, top_k):
    similarities = F.cosine_similarity(val_embedding, train_embeddings, dim=1)
    top_indices = torch.topk(similarities, k=top_k).indices
    
    return [(similarities[idx].item(), train_images[idx], train_masks[idx]) 
            for idx in top_indices]
    
def _shape_arrangement(images, arrange_mode):
    arranged = [None] * len(images)
    left, right = 0, len(images) - 1
    
    for i, img in enumerate(images):
        pos = left if i % 2 == 0 else right
        arranged[pos] = img
        left += (i % 2 == 0)
        right -= (i % 2 != 0)
        
    return arranged

def _insert_query_images(images, top, step, query_image, query_mask):
    if top==step:
        return images + [('Q', query_image.squeeze(0).permute(2, 0, 1).cpu().numpy(), 
                         query_mask.squeeze().cpu().numpy())]
    
    if step > 0 and step < len(images):
        result = []
        for i, img in enumerate(images):
            result.append(img)
            if (i + 1) % step == 0 or i == len(images) - 1:
                result.append(('Q', query_image.squeeze(0).permute(2, 0, 1), 
                              query_mask.squeeze()))
        return result
    
    raise ValueError(f"Invalid Step: {step}")

def arrange_images(images, arrange_mode, top, step, query_image, query_mask):
    if arrange_mode == 'random':
        random.shuffle(images)
    elif arrange_mode == 'descend':
        images.sort(key=lambda x: x[0], reverse=True)
    elif arrange_mode == 'ascend':
        images.sort(key=lambda x: x[0], reverse=False)
    elif arrange_mode.endswith('_shape'):
        images.sort(key=lambda x: x[0], reverse=(arrange_mode == 'u_shape'))
        images = _shape_arrangement(images, arrange_mode)
    
    return _insert_query_images(images, top, step, query_image, query_mask)


def save_image_data(data, save_path, is_mask=False):
    if isinstance(data, torch.Tensor):
        data = data.cpu().byte()
        if data.ndim == 3 and data.shape[-1] == 3:
            data = data.permute(2, 0, 1)
        data = ToPILImage()(data)
    else:
        if data.shape[0] == 3 and len(data.shape) == 3:
            data = data.transpose(1, 2, 0)
        data = Image.fromarray(data.astype(np.uint8)).convert('L' if is_mask else 'RGB')
    
    data.save(save_path)

def save_results(arranged_images, save_dir, video_name):
    paths = {
        'images': os.path.join(save_dir, 'images', video_name),
        'masks': os.path.join(save_dir, 'masks', video_name),
        # 'gt_masks': os.path.join(save_dir, 'gt_masks', video_name)
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    for i, (similarity, image, mask) in enumerate(arranged_images):
        save_image_data(image, os.path.join(paths['images'], f"{i + 1:05d}.jpg"))
        if similarity != 'Q':
            save_image_data(mask, os.path.join(paths['masks'], f"{i + 1:05d}.png"), is_mask=True)

def get_per_obj_prompt(mask, add_condition):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    palette = Image.fromarray(mask).getpalette()
    
    obj_ids = np.unique(mask)[1:].tolist()
    
    if add_condition == 'mask':
        return {obj_id: (mask == obj_id) for obj_id in obj_ids}, palette
        
    elif add_condition in ['box', 'point']:
        x, y, w, h = cv2.boundingRect(mask)
        
        if add_condition == 'box':
            box = np.array([x, y, x + w, y + h])
            return {obj_id: box for obj_id in obj_ids}, palette
        else:  # 'point'
            point = np.array([[x + w // 2, y + h // 2]])
            return {obj_id: point for obj_id in obj_ids}, palette
            
    else:
        raise ValueError(f"Invalid add_condition: {add_condition}. Please use 'mask', 'box' or 'point'")


def combine_masks(per_obj_mask, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj_id in sorted(per_obj_mask, reverse=True):
        mask[per_obj_mask[obj_id].reshape(height, width)] = obj_id
    return mask
    
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(predictor, arranged_images, video_dir, output_dir, video_name, add_condition, score_thresh=0.0):
    state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
    height, width = state["video_height"], state["video_width"]
    query_frames = []

    # Add initial prompts
    for frame_idx, (similarity, image, mask) in enumerate(arranged_images):
        if similarity != 'Q':
            per_obj_mask, input_palette = get_per_obj_prompt(mask, add_condition)
            for obj_id, obj_prompt in per_obj_mask.items():
                if add_condition == 'mask':
                    predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=obj_prompt,
                    )
                elif add_condition == 'box':
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        box=obj_prompt,
                    )
                elif add_condition == 'point':
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=obj_prompt,
                        labels=np.array([1], np.int32),
                    )
                else:
                    raise ValueError(f"Invalid add_condition: {add_condition}")
        else:
            query_frames.append(frame_idx)

    # Process frames in parallel
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx not in query_frames:
                continue
                
            futures.append(executor.submit(
                process_frame,
                frame_idx, obj_ids, mask_logits,
                output_dir, video_name, height, width, score_thresh
            ))
        
        for future in futures:
            results.update(future.result())
    
    output_palette = input_palette or DAVIS_PALETTE  
    return results, output_palette

def process_frame(frame_idx, obj_ids, mask_logits, output_dir, video_name, height, width, score_thresh):
    masks = {
        obj_id: (mask_logits[i] > score_thresh).cpu().numpy()
        for i, obj_id in enumerate(obj_ids)
    }
    return {frame_idx: combine_masks(masks, height, width)}

def calculate_metrics(pred_mask, gt_mask):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if len(np.unique(gt_mask)) > 2:
        gt_mask = np.where(gt_mask > 128, 255, 0)
    if len(np.unique(pred_mask)) > 2:
        pred_mask = np.where(pred_mask > 128, 255, 0)
    metrics = {
        'iou': binary.jc(pred_mask, gt_mask),
        'dice': binary.dc(pred_mask, gt_mask),
        'precision': binary.precision(pred_mask, gt_mask),
        'recall': binary.recall(pred_mask, gt_mask),
        'specificity': binary.specificity(pred_mask, gt_mask)
    }
    return metrics

def test_with_best_hyper(args, train_loader, test_loader, WEIGHT_FILES, predictor, best_hyper, device):
    top, step, weight_name, best_frame = best_hyper
    weight_path = next(w for w in WEIGHT_FILES if weight_name in w)
    model = load_model(weight_path, device)
    train_embeddings, train_images, train_masks, train_image_paths = extract_embeddings(model, train_loader, device)

    result_dir = os.path.join(args.result_save_dir, 'test', f'top{top}_step{step}_{weight_name}')
    
    metrics = {
        'iou': [], 'dice': [], 'precision': [], 
        'recall': [], 'specificity': []
    }
    
    for idx, (images, np_images, np_masks, img_path) in enumerate(test_loader):
        video_name = os.path.splitext(os.path.basename(img_path[0]))[0]
        embedding = model(images.to(device))
        
        top_images = get_top_k_similarities(embedding, train_embeddings, train_images, train_masks, top)
        arranged = arrange_images(top_images, args.arrange, top, step, np_images, np_masks)
        
        save_results(arranged, result_dir, video_name)
        masks, output_palette = vos_inference(predictor, arranged, os.path.join(result_dir, 'images', video_name), result_dir, video_name, args.add_condition)
        
        if best_frame in masks:
            m = calculate_metrics(masks[best_frame], arranged[best_frame][2])
            for k in metrics:
                metrics[k].append(m[k])
        else:
            ValueError(f"Best frame {best_frame} not found in masks")
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    log_test_results(args.result_save_dir, best_hyper, avg_metrics)

def log_test_results(save_dir, best_hyper, metrics):
    with open(os.path.join(save_dir, 'hyperparameters_results.txt'), 'a') as f:
        f.write(f"\nTest Results - Best hyperparameters:\n")
        f.write(f"top_k={best_hyper[0]}, step={best_hyper[1]}, model={best_hyper[2]}, LOC={best_hyper[3]+1}\n")
        f.write(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}, Specificity: {metrics['specificity']:.4f}\n")
    
    print(f"\nTest Results - Best hyperparameters:")
    print(f"top_k={best_hyper[0]}, step={best_hyper[1]}, model={best_hyper[2]}, LOC={best_hyper[3]+1}")
    print(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}, Specificity: {metrics['specificity']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Medical Image Similarity Search and Segmentation")
    parser.add_argument("--sam2_cfg", type=str, required=True, help="Path to SAM 2 config file")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM 2 checkpoint")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--result_save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--arrange", type=str, default="descend", help="Arrange mode for images")
    parser.add_argument('--add_condition', type=str, default='mask', help="Add condition for VOS")
    parser.add_argument("--apply_postprocessing", action="store_true", help="Apply postprocessing to masks")
    parser.add_argument("--use_vos_optimized_video_predictor", action="store_true", help="Use VOS-optimized predictor")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _, test_loader = load_datasets(args.train_dir, args.val_dir, args.test_dir)
    
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
        vos_optimized=args.use_vos_optimized_video_predictor,
    )
    
    best_hyper = (5, 5, 'sscd_imagenet_advanced', 5)  # top=5, step=5, loc_6

    test_with_best_hyper(args, train_loader, test_loader, WEIGHT_FILES, predictor, best_hyper, device)

if __name__ == "__main__":
    # Set random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    main()
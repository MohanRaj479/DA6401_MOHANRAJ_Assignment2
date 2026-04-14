"""
Main training script for computer vision tasks: 
Classification, Bounding Box Regression, and Semantic Segmentation.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import wandb
from sklearn.metrics import f1_score
import numpy as np

# Internal imports for the specific architectures
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

# Global storage for hooking into intermediate layers
layer_outputs = {}

def capture_feature_maps(layer_id):
    """Callback function to grab activations during the forward pass."""
    def internal_hook(module, input_tensor, output_tensor):
        layer_outputs[layer_id] = output_tensor.detach()
    return internal_hook

def setup_model_weights(module):
    """Sets up the initial weights using common heuristics for deep nets."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def prepare_data_streams(data_path, sz_batch=32):
    """Sets up the loaders for both training and validation."""
    # Standard ImageNet stats for normalization
    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)
    
    # Define how we warp/clean our images
    augmentor = alb.Compose([
        alb.Resize(224, 224),
        alb.HorizontalFlip(p=0.5),
        alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        alb.Normalize(mean=rgb_mean, std=rgb_std),
        ToTensorV2()
    ], bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    eval_processor = alb.Compose([
        alb.Resize(224, 224),
        alb.Normalize(mean=rgb_mean, std=rgb_std),
        ToTensorV2()
    ], bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # Initialize the actual dataset objects
    ds_train = OxfordIIITPetDataset(data_path, split="train", transforms=augmentor)
    ds_valid = OxfordIIITPetDataset(data_path, split="test", transforms=eval_processor)
    
    loader_train = DataLoader(ds_train, batch_size=sz_batch, shuffle=True, num_workers=2)
    loader_valid = DataLoader(ds_valid, batch_size=sz_batch, shuffle=False, num_workers=2)
    
    return loader_train, loader_valid

def convert_to_centroid(coords):
    """Switches bounding box from [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x_min, y_min, x_max, y_max = coords.unbind(1)
    return torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min], dim=1)

def convert_to_corners(coords):
    """Switches bounding box from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = coords.unbind(1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

def log_visual_boxes(imgs, p_boxes, t_boxes, scores):
    """Utility to push images with overlaid boxes to W&B."""
    viz_list = []
    # Hardcoded denormalization stats
    m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(imgs.device)
    s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(imgs.device)
    
    for idx in range(min(10, imgs.shape[0])):
        # Reverse normalization for viewing
        raw_img = (imgs[idx] * s + m).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        
        pred_coords = p_boxes[idx].cpu().numpy()
        true_coords = t_boxes[idx].cpu().numpy()
        
        viz_list.append(wandb.Image(raw_img, boxes={
            "predictions": {
                "box_data": [{"position": {"minX": pred_coords[0], "minY": pred_coords[1], "maxX": pred_coords[2], "maxY": pred_coords[3]}, "class_id": 1}],
                "class_labels": {1: f"IoU: {scores[idx]:.2f}"}
            },
            "ground_truth": {
                "box_data": [{"position": {"minX": true_coords[0], "minY": true_coords[1], "maxX": true_coords[2], "maxY": true_coords[3]}, "class_id": 2}],
                "class_labels": {2: "GT"}
            }
        }))
    return viz_list

# --- TASK 1: CLASSIFICATION ---
def train_classifier(args, device, train_loader, val_loader):
    wandb.init(project="DA6401_Assignment II_update", name=f"classification_bn_{args.use_bn}_dropout_{args.dropout}", config=vars(args))
    
    net = VGG11Classifier(num_classes=37, dropout_p=args.dropout, use_bn=args.use_bn).to(device)
    net.apply(setup_model_weights)
    
    # Hook into the middle of the encoder to see what the kernels are learning
    target = net.encoder.block3[0][0] if args.use_bn else net.encoder.block3[0]
    target.register_forward_hook(capture_feature_maps('conv3_activations'))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_genie = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    
    top_score = 0.0
    stagnation_counter = 0

    for curr_ep in range(args.epochs):
        net.train()
        running_loss = 0.0
        for data in train_loader:
            x, y = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            preds = net(x)
            batch_loss = loss_fn(preds, y)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            running_loss += batch_loss.item()

        net.eval()
        v_loss, preds_list, labels_list = 0.0, [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                x_val, y_val = batch['image'].to(device), batch['label'].to(device)
                logits = net(x_val)
                v_loss += loss_fn(logits, y_val).item()
                preds_list.extend(logits.argmax(1).cpu().numpy())
                labels_list.extend(y_val.cpu().numpy())
                
                if i == 0 and 'conv3_activations' in layer_outputs:
                    wandb.log({"conv3_activations": wandb.Histogram(layer_outputs['conv3_activations'].cpu().numpy())}, commit=False)

        current_f1 = f1_score(labels_list, preds_list, average='macro', zero_division=0)
        lr_genie.step(current_f1)
        
        wandb.log({"epoch": curr_ep, "train_loss": running_loss/len(train_loader), "val_loss": v_loss/len(val_loader), "val_macro_f1": current_f1})
        print(f"Ep {curr_ep+1} | Macro-F1: {current_f1:.4f}")

        if current_f1 > top_score:
            top_score = current_f1
            stagnation_counter = 0
            torch.save(net.state_dict(), "checkpoints/classifier.pth")
        else:
            stagnation_counter += 1
            if stagnation_counter >= 15: break
            
    wandb.finish()

# --- TASK 2: LOCALIZATION ---
def train_localization(args, device, train_loader, val_loader):
    wandb.init(project="DA6401_Assignment II", name="localization", config=vars(args))
    
    net = VGG11Localizer(in_channels=3).to(device)
    net.apply(setup_model_weights)
    
    regression_loss = nn.SmoothL1Loss()
    overlap_loss = IoULoss(reduction="none")
    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
    lr_genie = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=15)
    
    peak_iou = 0.0

    for curr_ep in range(args.epochs):
        net.train()
        total_train_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            gt_xyxy = batch['bbox'].to(device)
            gt_center = convert_to_centroid(gt_xyxy)
            
            optimizer.zero_grad()
            out_center = net(imgs)
            out_xyxy = convert_to_corners(out_center)
            
            # Combine Euclidean distance of centers with the actual overlap loss
            combined_loss = regression_loss(out_center, gt_center) + (50.0 * overlap_loss(out_xyxy, gt_xyxy).mean())
            combined_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_train_loss += combined_loss.item()

        net.eval()
        v_loss, running_iou_sum = 0.0, 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                imgs, gt_xyxy = batch['image'].to(device), batch['bbox'].to(device)
                out_center = net(imgs)
                out_xyxy = convert_to_corners(out_center)
                
                batch_iou_err = overlap_loss(out_xyxy, gt_xyxy)
                v_loss += (regression_loss(out_center, convert_to_centroid(gt_xyxy)) + (50.0 * batch_iou_err.mean())).item()
                running_iou_sum += batch_iou_err.mean().item()

                if i == 0:
                    wandb.log({"Localization_Edge_Cases": log_visual_boxes(imgs, out_xyxy, gt_xyxy, 1.0 - batch_iou_err)}, commit=False)

        mean_iou = 1.0 - (running_iou_sum / len(val_loader))
        lr_genie.step(mean_iou)
        wandb.log({"epoch": curr_ep, "train_loss": total_train_loss/len(train_loader), "val_loss": v_loss/len(val_loader), "val_iou": mean_iou})
        print(f"Ep {curr_ep+1} | Mean IoU: {mean_iou:.4f}")

        if mean_iou > peak_iou:
            peak_iou = mean_iou
            torch.save(net.state_dict(), "checkpoints/localizer.pth")
            
    wandb.finish()

# --- TASK 3: SEGMENTATION ---
def train_segmentation(args, device, train_loader, val_loader):
    wandb.init(project="DA6401_Assignment II", name=f"segmentation_{args.freeze_mode}", config=vars(args))
    
    net = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout).to(device)
    net.apply(setup_model_weights)

    # Logic for freezing parts of the encoder
    if args.freeze_mode == "frozen":
        for p in net.encoder.parameters(): p.requires_grad = False
    elif args.freeze_mode == "partial":
        for tag, p in net.encoder.named_parameters():
            if "block5" not in tag: p.requires_grad = False
    
    # Background often dominates, so we down-weight the 'trimap' middle class
    penalties = torch.tensor([1.0, 0.2, 1.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=penalties)
    
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    lr_genie = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    
    best_dice_val = 0.0
    stall_count = 0

    for curr_ep in range(args.epochs):
        net.train()
        train_running_loss = 0.0
        for batch in train_loader:
            img, mask = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            mask_preds = net(img)
            loss = loss_fn(mask_preds, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            train_running_loss += loss.item()

        net.eval()
        v_loss, total_dice, total_pixel_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                img, mask = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
                res = net(img)
                v_loss += loss_fn(res, mask).item()

                hard_preds = res.argmax(1)
                total_pixel_acc += (hard_preds == mask).float().mean().item()

                # Manual Dice calculation for the 3 classes
                for cls_idx in range(3):
                    p_bin, m_bin = (hard_preds == cls_idx), (mask == cls_idx)
                    inter = (p_bin & m_bin).sum().float()
                    denom = p_bin.sum() + m_bin.sum()
                    total_dice += (2. * inter / (denom + 1e-7)).item()
                    
        final_dice = total_dice / (len(val_loader) * 3)
        final_acc = total_pixel_acc / len(val_loader)
        
        lr_genie.step(final_dice)
        wandb.log({"epoch": curr_ep, "train_loss": train_running_loss/len(train_loader), "val_loss": v_loss/len(val_loader), "val_dice": final_dice, "val_pixel_acc": final_acc})
        print(f"Ep {curr_ep+1} | Dice: {final_dice:.4f} | Acc: {final_acc:.4f}")

        if final_dice > best_dice_val:
            best_dice_val = final_dice
            stall_count = 0
            torch.save(net.state_dict(), "checkpoints/unet.pth")
        else:
            stall_count += 1
            if stall_count >= 15: break
            
    wandb.finish()

def parse_boolean(item):
    """Helper to handle bools from command line better."""
    if isinstance(item, bool): return item
    return item.lower() in ('true', '1', 't', 'y', 'yes')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task Vision Training")
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--data_dir", type=str, default="/content/oxford-iiit-pet") 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_bn", type=parse_boolean, nargs='?', const=True, default=True)
    parser.add_argument("--freeze_mode", type=str, default="none", choices=["none", "frozen", "partial"])

    config = parser.parse_args()
    computing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        
    t_load, v_load = prepare_data_streams(config.data_dir, config.batch_size)

    # Route to the specific sub-training routine
    if config.task == "classification":
        train_classifier(config, computing_device, t_load, v_load)
    elif config.task == "localization":
        train_localization(config, computing_device, t_load, v_load)
    elif config.task == "segmentation":
        train_segmentation(config, computing_device, t_load, v_load)
        
# """Training entrypoint"""
# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import wandb
# from sklearn.metrics import f1_score
# import numpy as np

# from data.pets_dataset import OxfordIIITPetDataset
# from models.classification import VGG11Classifier
# from models.localization import VGG11Localizer
# from models.segmentation import VGG11UNet
# from losses.iou_loss import IoULoss

# activation_cache = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation_cache[name] = output.detach()
#     return hook

# def init_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

# def get_dataloaders(root_dir: str, batch_size: int = 32):
#     mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
#     train_transform = A.Compose([
#         A.Resize(224, 224), A.HorizontalFlip(p=0.5),
#         A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
#         A.Normalize(mean=mean, std=std), ToTensorV2()
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

#     val_transform = A.Compose([
#         A.Resize(224, 224), A.Normalize(mean=mean, std=std), ToTensorV2()
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

#     train_loader = DataLoader(OxfordIIITPetDataset(root_dir, split="train", transforms=train_transform), batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(OxfordIIITPetDataset(root_dir, split="test", transforms=val_transform), batch_size=batch_size, shuffle=False, num_workers=2)
#     return train_loader, val_loader

# def voc_to_cxcywh(bboxes):
#     x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
#     return torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=1)

# def cxcywh_to_voc(bboxes):
#     cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
#     return torch.stack([cx - (w / 2.0), cy - (h / 2.0), cx + (w / 2.0), cy + (h / 2.0)], dim=1)

# def draw_bboxes_wandb(images, pred_boxes, target_boxes, ious):
#     wandb_images = []
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
    
#     for i in range(min(10, images.size(0))):
#         img = images[i] * std + mean
#         img = torch.clamp(img, 0, 1).cpu().numpy().transpose(1, 2, 0)
        
#         p_x1, p_y1, p_x2, p_y2 = pred_boxes[i].cpu().numpy()
#         t_x1, t_y1, t_x2, t_y2 = target_boxes[i].cpu().numpy()
#         iou_val = ious[i].item()
        
#         wandb_images.append(wandb.Image(img, boxes={
#             "predictions": {"box_data": [{"position": {"minX": p_x1, "minY": p_y1, "maxX": p_x2, "maxY": p_y2}, "class_id": 1, "domain": "pixel"}], "class_labels": {1: f"Pred IoU:{iou_val:.2f}"}},
#             "ground_truth": {"box_data": [{"position": {"minX": t_x1, "minY": t_y1, "maxX": t_x2, "maxY": t_y2}, "class_id": 2, "domain": "pixel"}], "class_labels": {2: "Target"}}
#         }))
#     return wandb_images

# #  CLASSIFICATION
# def train_classifier(args, device, train_loader, val_loader):
#     wandb.init(project="DA6401_Assignment II_update", name=f"classification_bn_{args.use_bn}_dropout_{args.dropout}", config=vars(args))
#     model = VGG11Classifier(num_classes=37, dropout_p=args.dropout, use_bn=args.use_bn).to(device)
#     model.apply(init_weights)
    
#     target_layer = model.encoder.block3[0][0] if args.use_bn else model.encoder.block3[0]
#     target_layer.register_forward_hook(get_activation('conv3_activations'))
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#     best_f1 = 0.0; epochs_no_improve = 0

#     for epoch in range(args.epochs):
#         model.train(); train_loss = 0.0
#         for batch in train_loader:
#             images, labels = batch['image'].to(device), batch['label'].to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(images), labels)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             train_loss += loss.item()

#         model.eval(); val_loss, all_preds, all_labels = 0.0, [], []
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(val_loader):
#                 images, labels = batch['image'].to(device), batch['label'].to(device)
#                 outputs = model(images)
#                 val_loss += criterion(outputs, labels).item()
#                 all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#                 if batch_idx == 0 and 'conv3_activations' in activation_cache:
#                     wandb.log({"conv3_activations": wandb.Histogram(activation_cache['conv3_activations'].cpu().numpy())}, commit=False)

#         macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#         scheduler.step(macro_f1)
#         wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_macro_f1": macro_f1})
#         print(f"Epoch {epoch+1} - Macro F1: {macro_f1:.4f}")

#         if macro_f1 > best_f1:
#             best_f1 = macro_f1; epochs_no_improve = 0
#             torch.save({"state_dict": model.state_dict()}, "checkpoints/classifier.pth")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= 15: break
#     wandb.finish()

# # LOCALIZATION 
# def train_localization(args, device, train_loader, val_loader):
#     wandb.init(project="DA6401_Assignment II", name="localization", config=vars(args))
#     model = VGG11Localizer(in_channels=3).to(device)
#     model.apply(init_weights) 
    
#     criterion_reg = nn.SmoothL1Loss(); criterion_iou = IoULoss(reduction="none")
#     optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
#     best_iou = 0.0; epochs_no_improve = 0

#     for epoch in range(args.epochs):
#         model.train(); train_loss = 0.0
#         for batch in train_loader:
#             images = batch['image'].to(device)
#             bboxes_xyxy = batch['bbox'].to(device) 
#             bboxes_cxcywh = voc_to_cxcywh(bboxes_xyxy)
            
#             optimizer.zero_grad()
#             outputs_cxcywh = model(images)
#             outputs_xyxy = cxcywh_to_voc(outputs_cxcywh)
            
#             loss = criterion_reg(outputs_cxcywh, bboxes_cxcywh) + (50.0 * criterion_iou(outputs_xyxy, bboxes_xyxy).mean())
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             train_loss += loss.item()

#         model.eval(); val_loss, val_iou_loss = 0.0, 0.0
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(val_loader):
#                 images = batch['image'].to(device)
#                 bboxes_xyxy = batch['bbox'].to(device)
#                 bboxes_cxcywh = voc_to_cxcywh(bboxes_xyxy)
                
#                 outputs_cxcywh = model(images)
#                 outputs_xyxy = cxcywh_to_voc(outputs_cxcywh)
                
#                 l_iou_batch = criterion_iou(outputs_xyxy, bboxes_xyxy)
#                 l_iou = l_iou_batch.mean()
#                 val_loss += (criterion_reg(outputs_cxcywh, bboxes_cxcywh) + (50.0 * l_iou)).item()
#                 val_iou_loss += l_iou.item()

#                 if batch_idx == 0:
#                     ious_scores = 1.0 - l_iou_batch
#                     wandb.log({"Localization_Edge_Cases": draw_bboxes_wandb(images, outputs_xyxy, bboxes_xyxy, ious_scores)}, commit=False)

#         avg_val_iou = 1.0 - (val_iou_loss / len(val_loader))
#         scheduler.step(avg_val_iou)
#         wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_iou": avg_val_iou})
#         print(f"Epoch {epoch+1} - Val IoU: {avg_val_iou:.4f}")

#         if avg_val_iou > best_iou:
#             best_iou = avg_val_iou; epochs_no_improve = 0
#             torch.save({"state_dict": model.state_dict()}, "checkpoints/localizer.pth")
#     wandb.finish()

# # SEGMENTATION
# def train_segmentation(args, device, train_loader, val_loader):
#     wandb.init(project="DA6401_Assignment II", name=f"segmentation_{args.freeze_mode}", config=vars(args))
#     model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout).to(device)
#     model.apply(init_weights)

#     if args.freeze_mode == "frozen":
#         for param in model.encoder.parameters():
#             param.requires_grad = False
#     elif args.freeze_mode == "partial":
#         for name, param in model.encoder.named_parameters():
#             if "block5" not in name: 
#                 param.requires_grad = False
    
#     class_weights = torch.tensor([1.0, 0.2, 1.0]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#     best_dice = 0.0; epochs_no_improve = 0

#     for epoch in range(args.epochs):
#         model.train(); train_loss = 0.0
#         for batch in train_loader:
#             images, masks = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
#             optimizer.zero_grad()
#             loss = criterion(model(images), masks)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             train_loss += loss.item()

#         model.eval(); val_loss, dice_score, pixel_acc = 0.0, 0.0, 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 images, masks = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
#                 outputs = model(images)
#                 val_loss += criterion(outputs, masks).item()

#                 preds = torch.argmax(outputs, dim=1)
                
#                 pixel_acc += (preds == masks).sum().float().item() / masks.numel()

#                 for c in range(3):
#                     pred_c = (preds == c)
#                     mask_c = (masks == c)
#                     intersection = (pred_c & mask_c).sum().float()
#                     union = pred_c.sum() + mask_c.sum()
#                     dice_score += (2. * intersection / (union + 1e-6)).item()
                    
#         dice_score /= (len(val_loader) * 3)
#         pixel_acc /= len(val_loader)
        
#         scheduler.step(dice_score)
#         wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_dice": dice_score, "val_pixel_acc": pixel_acc})
#         print(f"Epoch {epoch+1} - Val Dice: {dice_score:.4f} | Val Pixel Acc: {pixel_acc:.4f}")

#         if dice_score > best_dice:
#             best_dice = dice_score; epochs_no_improve = 0
#             torch.save({"state_dict": model.state_dict()}, "checkpoints/unet.pth")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= 15: break
#     wandb.finish()

# def str2bool(v):
#     if isinstance(v, bool): return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
#     return False

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", type=str, default="classification")
#     parser.add_argument("--data_dir", type=str, default="/content/oxford-iiit-pet") 
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--dropout", type=float, default=0.5)
#     parser.add_argument("--use_bn", type=str2bool, nargs='?', const=True, default=True)
#     parser.add_argument("--freeze_mode", type=str, default="none", choices=["none", "frozen", "partial"])

#     args = parser.parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs("checkpoints", exist_ok=True)
#     train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

#     if args.task == "classification": train_classifier(args, device, train_loader, val_loader)
#     elif args.task == "localization": train_localization(args, device, train_loader, val_loader)
#     elif args.task == "segmentation": train_segmentation(args, device, train_loader, val_loader)

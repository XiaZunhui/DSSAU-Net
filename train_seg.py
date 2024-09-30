
import argparse
from utils.dataset_segmentation import *
from utils.augmentation import *
import random
from utils.criterion import MyCriterion
from torch.utils.data import DataLoader, random_split
from utils.metric import DSC
from Mynet.DSSAU_Net import DSSAU_Net
from utils.evaluator import Segment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'

def get_model():
    model = DSSAU_Net(seg_num_classes=3)
    return model


def get_dataloader(dir, img_size=512, batch_size=2):
    tf_train = JointTransform2D(img_size=img_size, low_img_size=128, ori_size=img_size, crop=None, p_flip=0.5,
                                p_rota=0.5, p_scale=0.0, p_gaussn=0.0, p_contr=0.0, p_gama=0.0, p_distor=0.0,
                                color_jitter_params=None,
                                long_mask=True)
    tf_val = JointTransform2D(img_size=img_size, low_img_size=128, ori_size=img_size, crop=None, p_flip=0.0,
                              p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0, p_gama=0.0, p_distor=0.0,
                              color_jitter_params=None,
                              long_mask=True)


    full_dataset = DatasetSegmentation(dir, None)


    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_set.dataset.transform = tf_train
    val_set.dataset.transform = tf_val

    # 创建DataLoader
    dataset_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataset_val = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return dataset_train, dataset_val


def save_training_process(train_losses, val_losses):
    with open("./process.txt", "w") as file:
        num_epochs = len(train_losses)
        for i in range(num_epochs):
            file.write(f"Epoch {i} Train Loss: {train_losses[i]}")
            if (i + 1) >= (num_epochs // 2):
                index = int((num_epochs // 2) - 1)
                file.write(f"Epoch {i} Val Loss: {val_losses[index]}")
            file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/xiazunhui/project/IUGC/data/dataset_sample',
                        help='the path of images')
    parser.add_argument('--BATCH_SIZE', type=int, default=20, help='batch size')
    parser.add_argument('--IMG_SIZE', type=int, default=256, help='image size')
    parser.add_argument('--NUM_EPOCHS', type=int, default=10, help='epoch')
    parser.add_argument('--BASE_LEARNING_RATE', type=float, default=0.0001, help='learning rate')

    args = parser.parse_args()
    # ========================== set random seed ===========================#
    seed_value = 2024  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    BATCH_SIZE = args.BATCH_SIZE
    NUM_EPOCHS = args.NUM_EPOCHS
    BASE_LEARNING_RATE = args.BASE_LEARNING_RATE
    IMG_SIZE = args.IMG_SIZE
    data_dir = os.path.join(args.data_dir, 'pos')

    # ========================== get model, dataloader, optimizer and so on =========================#
    model = get_model()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LEARNING_RATE,
                                  betas=(0.9, 0.999), weight_decay=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_from()

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    # model.load_state_dict(st)
    dataset_train, dataset_val = get_dataloader(data_dir, IMG_SIZE, BATCH_SIZE)
    segment = Segment()
    criterion = MyCriterion()  # combined loss function
    evalue = DSC()  # metric to find best model
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_val_score = 0
    train_losses = []
    val_scores = []
    avg_ten_loss = []
    val_losses = []
    ce_loss = []
    dice_loss = []
    # ========================== training =============================#
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, sample in enumerate(dataset_train):
            # print(f"Epoch[{epoch + 1}/{NUM_EPOCHS}] | Batch {batch_idx}: ", end="")
            batch_train_loss = []
            imgs = sample['image'].to(dtype=torch.float32, device=device)  # (BS, 3, 512, 512)
            masks = sample['label'].to(device=device).squeeze(1)  # (BS,512,512)   0: background 1:ps 2:fh
            # low_masks = sample['low_label'].to(device=device).squeeze(1)  # maybe it's not important

            # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            pred = model(imgs)  # (BS, 3, 512, 512) maybe the resolution is not (512, 512), never mind, you need to keep the same resolution of label and pred.
            # print(masks.shape,pred.shape)
            train_loss, dice, ce = criterion(pred, masks)

            # scaler.scale(train_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
            # ===============================================
            current_lr = optimizer.param_groups[0]['lr']

            batch_train_loss.append(train_loss.detach().cpu().numpy())
            avg_ten_loss.append(train_loss.detach().cpu().numpy())
            ce_loss.append(ce.detach().cpu().numpy())
            dice_loss.append(dice.detach().cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                avg_loss_last_10 = np.mean(avg_ten_loss)
                ce_ten = np.mean(ce_loss)
                dice_ten = np.mean(dice_loss)
                print(
                    f"Train:Epoch[{epoch + 1}/{NUM_EPOCHS}] | Batch [{batch_idx + 1}/{len(dataset_train)}]:  Loss: {avg_loss_last_10:.4f} | ce_loss:{ce_ten:.4f} | dice_loss:{dice_ten:.4f} | lr:{current_lr:.6f}")
                avg_ten_loss = []  # 重置列表，为下一个10个批次做准备

        # print("================================")
        # scheduler.step()
        train_losses.append(np.mean(batch_train_loss))
        print(f'Total Loss:{np.mean(train_losses)}')

        # ========================= validation =======================#
        if (epoch + 1) >= 1:  # (NUM_EPOCHS // 2):
            model.eval()
            val_score_ls = []
            with torch.no_grad():
                for batch_idx, sample in enumerate(dataset_val):
                    imgs = sample['image'].to(dtype=torch.float32, device=device)  # (BS,3,512,512)
                    masks = sample['label'].to(device=device).squeeze(1)  # (BS,512,512)
                    pred = model(imgs)
                    val_loss, _, _ = criterion(pred, masks)
                    val_score_one_batch = evalue(pred, masks)
                    # result = segment.evaluation(pred=pred, label=masks)
                    val_score_ls.append(val_score_one_batch.detach().cpu().numpy())
                    val_losses.append(val_loss.detach().cpu().numpy())

                val_score = np.mean(val_score_ls)
                # val_scores.append(val_score)
                print(
                    f"Val:Epoch[{epoch + 1}/{NUM_EPOCHS}] | Val_Loss:{np.mean(val_losses)} | Val_score:{val_score:.6f}")

                # ================  SAVING ================#
                if val_score > best_val_score:
                    best_val_score = val_score
                    checkpoint_dir = "./result"
                    # 直接创建目录，如果目录已存在，则不会抛出错误
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    path = f"{checkpoint_dir}/seg_best.pth"
                    if os.path.exists(path) and os.path.isfile(path):
                        os.remove(path)
                    # 保存模型的state_dict
                    torch.save(model.state_dict(), path)
                    print(f'new best score:{val_score}')
    # save_training_process(train_losses, val_scores)


if __name__ == '__main__':
    main()

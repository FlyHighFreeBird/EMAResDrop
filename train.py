import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from ResNetEMADropblock import ResNet50WithDropBlockAndEMA


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None,
                    loss_func=torch.nn.CrossEntropyLoss()):
    model.train()
    loss_function = loss_func
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 验证样本总个数
    total_num = len(data_loader.dataset)
    data_loader = tqdm(data_loader, file=sys.stdout)

    # 记录当前 epoch 的总损失
    epoch_loss = 0.0
    num_batches = 0

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # 更新平均损失

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

    # 计算平均损失
    avg_loss = epoch_loss / num_batches

    # 如果使用了 scheduler (例如 ReduceLROnPlateau)
    if scheduler is not None:
        # 根据当前的平均损失来调整学习率
        scheduler.step(avg_loss)

    model.eval()  # 切换到评估模式

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return avg_loss, sum_num.item() / total_num


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))  # 计算验证机的ＬＯＳＳ
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num, mean_loss.item()

def write_to_file(epoch, train_accuracy, train_loss, val_accuracy, val_loss, file_path):
    with open(file_path, "a") as f:
        f.write(f"Epoch {epoch}, Train Accuracy: {train_accuracy:.3f}, Train Loss: {train_loss:.3f}, "
                f"Val Accuracy: {val_accuracy:.3f}, Val Loss: {val_loss:.3f}\n")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据加载
    data_root = "data"
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                       transform=data_transform["val"])
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)
    print(train_dataset.classes)

    num_classes = 5

    net = ResNet50WithDropBlockAndEMA(num_classes=num_classes)
    net.to(device)
    net.load_state_dict(torch.load("best_dropout_EMA_226.pth", map_location=device))
    print("可训练参数:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    print(f"总参数数: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    print(f"可训练参数数: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.2f}M")

    loss_function = nn.CrossEntropyLoss()
    weight_decay = 0
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=0.001, weight_decay=weight_decay)

    epochs = 100
    best_acc = 0.0
    output_dir = "./"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file_path = os.path.join(output_dir, "training_log.txt")

    # 训练和验证
    for epoch in range(epochs):
        mean_loss, train_ac = train_one_epoch(model=net,
                                              optimizer=optimizer,
                                              data_loader=train_loader,
                                              device=device,
                                              epoch=epoch,
                                              scheduler=None,
                                              loss_func=loss_function)
        val_ac, val_lss = evaluate(model=net,
                                   data_loader=val_loader,
                                   device=device)
        print(f"[epoch {epoch}] train_accuracy: {train_ac:.3f}")
        print(f"[epoch {epoch}] train_loss: {mean_loss:.3f}")
        print(f"[epoch {epoch}] val_accuracy: {val_ac:.3f}")
        print(f"[epoch {epoch}] val_loss: {val_lss:.3f}")

        write_to_file(epoch, train_ac, mean_loss, val_ac, val_lss, log_file_path)

        if val_ac > best_acc:
            best_acc = val_ac
            torch.save(net.state_dict(), os.path.join(output_dir, "best.pth"))

    print(f'Finished Training, best acc is {best_acc:.3f}')


if __name__ == '__main__':
    main()
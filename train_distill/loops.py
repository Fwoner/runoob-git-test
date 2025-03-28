from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import math
import tensorboard_logger as tb_logger
import os

def train(model, train_loader, test_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 动态生成子文件夹路径
    log_base_dir = 'F:/PythonProject/DistillationExercise/logs'
    model_base_dir = 'F:/PythonProject/DistillationExercise/models'
    # 获取模型名称
    model_name = model.__class__.__name__
    # 动态生成子文件夹路径
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 格式化日期和时间
    log_subdir = os.path.join(log_base_dir, f"{model_name}_{current_time}")  # 子文件夹路径
    model_subdir = os.path.join(model_base_dir, f"{model_name}_{current_time}")  # 模型子文件夹路径

    os.makedirs(log_subdir, exist_ok=True)  # 创建子文件夹
    os.makedirs(model_subdir, exist_ok=True)  # 创建模型子文件夹

    logger = tb_logger.Logger(logdir=log_subdir, flush_secs=2)  # 使用子文件夹路径
    # 更新模型保存路径
    model_save_path = os.path.join(model_subdir, 'best_model.pth')
    model.train()
    best_acc = 0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        test_acc = test(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print("Best model saved")
            logger.log_value('best_acc', best_acc, epoch+1)
            epochs_no_improve = 0  # Reset epochs_no_improve if there is an improvement
        else:
            epochs_no_improve += 1  # Increment epochs_no_improve if no improvement

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        logger.log_value('train_acc', test_acc, epoch+1)
        logger.log_value('train_loss', running_loss / len(train_loader), epoch+1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    print("Best accuracy:", best_acc)
    return best_acc, model_save_path

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_knowledge_distillation(teacher, student, train_loader, test_loader, epochs, learning_rate, T,  device):
    student_name = student.__class__.__name__
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    log_base_dir = 'F:/PythonProject/DistillationExercise/logs/distill'
    student_base_dir = 'F:/PythonProject/DistillationExercise/save/student_model'
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_subdir = os.path.join(log_base_dir, f"{student_name}_{current_time}")
    student_subdir = os.path.join(student_base_dir, f"{student_name}_{current_time}")
    os.makedirs(log_subdir, exist_ok=True)
    os.makedirs(student_subdir, exist_ok=True)
    student_save_path = os.path.join(student_subdir, 'best_model.pth')
    logger = tb_logger.Logger(logdir=log_subdir, flush_secs=2)
    best_acc = 0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # 余弦退火两个损失函数的权重
            # soft_target_loss_weight, ce_loss_weight = adjust_alpha_cosine(epoch)
            soft_target_loss_weight, ce_loss_weight = 0.9, 0.1
            logger.log_value('soft_target_loss_weight', soft_target_loss_weight, epoch+1)
            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        test_acc = test(student, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), student_save_path)
            print("Best model saved")
            logger.log_value('best_acc', best_acc, epoch+1)
            epochs_no_improve = 0  # Reset epochs_no_improve if there is an improvement
        else:
            epochs_no_improve += 1  # Increment epochs_no_improve if no improvement

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        logger.log_value('student_loss', running_loss / len(train_loader), epoch+1)
        logger.log_value('student_acc', test_acc, epoch+1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    return best_acc

def adjust_alpha_cosine(epoch, init_soft_target_loss_weight = 0.9,  max_epoch=20):
    """
    余弦退火权重衰减策略
    参数说明：
    - epoch: 当前训练轮次
    - opt: 参数对象
    - max_epoch: 总训练轮次（默认240）
    - initial_alpha: 初始alpha值（默认0.9）
    """
    # 余弦函数计算（范围从0到2π）前90个epoch不动
    epoch = epoch - 10
    max_epoch = max_epoch - 10
    ce_loss_weight = 1 - init_soft_target_loss_weight
    soft_target_loss_weight = init_soft_target_loss_weight * (math.cos(2 * math.pi * epoch / max_epoch) + 1) / 2
    if soft_target_loss_weight < 0.5:
        soft_target_loss_weight = 0.5
    return normalize_weights(soft_target_loss_weight, ce_loss_weight)  # 保持权重归一化


def normalize_weights(soft_target_loss_weight, ce_loss_weight):
    """
    归一化损失函数权重参数，使得 opt.gamma + opt.alpha + opt.beta = 1
    """
    total_weight = soft_target_loss_weight + ce_loss_weight
    if total_weight == 0:
        raise ValueError("Total weight of gamma, alpha, and beta should not be zero.")

    soft_target_loss_weight /= total_weight
    ce_loss_weight /= total_weight
    return soft_target_loss_weight, ce_loss_weight

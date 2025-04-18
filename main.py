import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from network.testNetwork import DeepNN, LightNN
from train_distill.loops import train, test, train_knowledge_distillation, dynamic_knowledge_distillation
from dataset.cifar10 import get_cifar10_dataloader
# from dataset.tiny_imagenet import get_tiny_imagenet_dataloader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_loader, test_loader = get_tiny_imagenet_dataloader()   # 加载tiny imagenet 数据集
    train_loader, test_loader = get_cifar10_dataloader()
    torch.manual_seed(42)
    save_path = "models/DeepNN_2025-03-24_11-22-23/best_model.pth"  # 保存到项目根目录下的 models 文件夹

    # imagenet修改分类数量
    # nn_deep = DeepNN(num_classes=200).to(device)
    # nn_light = LightNN(num_classes=200).to(device)
    nn_deep = DeepNN(num_classes=10).to(device)
    nn_deep.load_state_dict(torch.load(save_path))
    nn_deep.eval()

    # train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)   训练教师模型

    test_accuracy_deep = test(nn_deep, test_loader, device)
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")

    # torch.save(nn_deep.state_dict(), save_path)   保存模型

    # Instantiate the lightweight network:
    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)

    # 在蒸馏之前训练学生模型:
    # test_accuracy_light_ce = train(nn_light, train_loader, test_loader, epochs=20, learning_rate=0.001, device=device)
    # print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    # print(f"Student best accuracy: {test_accuracy_light_ce:.2f}%")

    # 默认蒸馏温度T为4, 固定参数或者cos蒸馏
    # best_acc = train_knowledge_distillation(teacher=nn_deep, student=nn_light, train_loader=train_loader, test_loader=test_loader, epochs=20,
    #                              learning_rate=0.001, T=4, device=device)
    # 两个可学习参数
        # 使用动态权重蒸馏
    best_acc = dynamic_knowledge_distillation(teacher=nn_deep, student=nn_light, train_loader=train_loader, 
                                            test_loader=test_loader, epochs=20,
                                            learning_rate=0.001, T=4, device=device)
    test_accuracy_light_ce_and_kd = best_acc
    # Compare the student test accuracy with and without the teacher, after distillation
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    # print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()


    '''打印教师和学生模型参数数量'''
    # Print the number of parameters in the deep and new light models
    # total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
    # print(f"DeepNN parameters: {total_params_deep}")
    # total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
    # print(f"LightNN parameters: {total_params_light}")


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

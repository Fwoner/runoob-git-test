import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from network.testNetwork import DeepNN, LightNN
from train_distill.loops import train, test, train_knowledge_distillation
from dataset.cifar10 import get_cifar10_dataloader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_cifar10_dataloader()
    torch.manual_seed(42)
    save_path = "models/best_model.pth"  # 保存到项目根目录下的 models 文件夹
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


    train(nn_light, train_loader, test_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)

    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

    # 默认蒸馏温度T为4
    train_knowledge_distillation(teacher=nn_deep, student=nn_light, train_loader=train_loader, test_loader=test_loader, epochs=10,
                                 learning_rate=0.001, T=4, device=device)
    test_accuracy_light_ce_and_kd = test(nn_light, test_loader, device)
    # Compare the student test accuracy with and without the teacher, after distillation
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
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

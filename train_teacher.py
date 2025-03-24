from datetime import datetime

import torch
from network.testNetwork import DeepNN
from train_distill.loops import train, test
from dataset.cifar10 import get_cifar10_dataloader
import os
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloader()
    torch.manual_seed(42)
    # 初始化教师模型
    nn_deep = DeepNN(num_classes=10).to(device)
    _, save_path = train(nn_deep, train_loader, test_loader, epochs=20, learning_rate=0.001, device=device)

    test_accuracy_deep = test(nn_deep, test_loader, device)
    print("教师模型训练完毕")
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    final_model_path = os.path.join(os.path.dirname(save_path), 'final_model.pth').replace('\\', '/')
    torch.save(nn_deep.state_dict(), final_model_path)
    print(f"保存模型到 {final_model_path}")


if __name__ == "__main__":
    main()
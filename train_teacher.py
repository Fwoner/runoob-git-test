import torch
from network.testNetwork import DeepNN
from train_distill.loops import train, test
from dataset.cifar10 import get_cifar10_dataloader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloader()
    torch.manual_seed(42)
    save_path = "models/teacher_model.pth"  # 保存到项目根目录下的 models 文件夹
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, test_loader, epochs=20, learning_rate=0.001, device=device)
    test_accuracy_deep = test(nn_deep, test_loader, device)
    print("教师模型训练完毕")
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    torch.save(nn_deep.state_dict(), save_path)


if __name__ == "__main__":
    main()
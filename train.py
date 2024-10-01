import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
import argparse
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.pytorch
import datetime

def train(model, train_loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    accuracy = 100. * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return running_loss / len(train_loader), accuracy, precision, recall, f1, cm

def validate(model, val_loader, criterion, device):
    model.eval() 
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100. * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return val_loss / len(val_loader), accuracy, precision, recall, f1, cm

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=args.train_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=args.val_data_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_name == "OptimizedResnet101":
        from model import OptimizedResNet101Classifier
        model = OptimizedResNet101Classifier(num_classes=len(args.classes), frozen_layers=args.frozen_layers).to(args.device)
    if args.model_name == "OptimizedResnet50":
        from model import OptimizedResNet50Classifier
        model = OptimizedResNet50Classifier(num_classes=len(args.classes), frozen_layers=args.frozen_layers).to(args.device)
    elif args.model_name == "Resnet101":
        from model import ResNet101Classifier
        model = ResNet101Classifier(num_classes=len(args.classes), frozen_layers=args.frozen_layers).to(args.device)
    elif args.model_name == "Resnet50":
        from model import ResNet50Classifier
        model = ResNet50Classifier(num_classes=len(args.classes), frozen_layers=args.frozen_layers).to(args.device)
    else:
        raise KeyError("No valid model name! Valid models: OptimizedResnet101, OptimizedResnet50, Resnet101, Resnet50")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum)

    best_val_acc = 0.0
    best_epoch = 0

    run_name = args.model_save_path.split("/")[-1] + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with mlflow.start_run(run_name = run_name):
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_epochs", args.num_epochs)
        mlflow.log_param("momentum", args.momentum)
        mlflow.log_param("Version", os.path.dirname(args.model_save_path))

        for epoch in range(args.num_epochs):
            print(f'Epoch [{epoch+1}/{args.num_epochs}]')

            train_loss, train_acc, train_precision, train_recall, train_f1, _ = train(model, train_loader, criterion, optimizer, args.device)
            val_loss, val_acc, val_precision, val_recall, val_f1, val_cm = validate(model, val_loader, criterion, args.device)

            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_precision", train_precision, step=epoch)
            mlflow.log_metric("train_recall", train_recall, step=epoch)
            mlflow.log_metric("train_f1_score", train_f1, step=epoch)

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)

            #mlflow.log_artifact(f"confusion_matrix_epoch_{epoch+1}.txt", val_cm)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.model_save_path, "best_model.pth"))
                print(f'New best model saved with accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')
                mlflow.pytorch.log_model(model, "best_model")

        print(f'Training completed. Best model was at epoch {best_epoch} with validation accuracy {best_val_acc:.2f}%')
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_param("best_val_acc", best_val_acc)


def get_args():
    parser = argparse.ArgumentParser(description="Model configuration parser")

    # Model name
    parser.add_argument(
        "--model_name",
        type=str,
        default="OptimizedResnet101",
        help="Name of the model to be used. Valid models: OptimizedResnet101, OptimizedResnet50, Resnet101, Resnet50"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs for training"
    )
    
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for the optimizer"
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default='',
        help="Path to the training dataset"
    )
    
    parser.add_argument(
        "--val_data_path",
        type=str,
        default='',
        help="Path to the validation dataset"
    )
        
    parser.add_argument(
        "--classes",
        type=str,
        nargs='+',
        default=[],
        help="List of class names"
    )
    
    parser.add_argument(
        "--frozen_layers",
        type=str,
        nargs='+',
        default=['conv1', 'layer1', 'layer2'],
        help="List of frozen layers"
    )
    
    parser.add_argument(
        "--model_save_path",
        type=str,
        default='',
        help="Path to save the trained model"
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda:1' if torch.cuda.is_available() else 'cpu',
        help="Device to use for training (cpu or cuda)"
    )
    
    args = parser.parse_args()
    os.makedirs(args.model_save_path, exist_ok=True)

    return args

if __name__ == "__main__":
    args = get_args()
    print(f"Model Name: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Momentum: {args.momentum}")
    print(f"Train Data Path: {args.train_data_path}")
    print(f"Validation Data Path: {args.val_data_path}")
    print(f"Classes: {args.classes}")
    print(f"Frozen Layers: {args.frozen_layers}")
    print(f"Model Save Path: {args.model_save_path}")
    print(f"Device: {args.device}")

    main()

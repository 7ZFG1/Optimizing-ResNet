import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ResNet101Classifier
# from hyperparameters import *
import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

def plot_and_save_confusion_matrix(cm, classes, output_path, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.close()

def test(model, test_loader, criterion, device, class_names):
    model.eval()  
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = 100. * correct / total

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    precision_class = precision_score(all_labels, all_predictions, average=None)
    recall_class = recall_score(all_labels, all_predictions, average=None)
    f1_class = f1_score(all_labels, all_predictions, average=None)

    precision_avg = precision_score(all_labels, all_predictions, average='weighted')
    recall_avg = recall_score(all_labels, all_predictions, average='weighted')
    f1_avg = f1_score(all_labels, all_predictions, average='weighted')

    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    return (avg_test_loss, avg_test_acc, conf_matrix, conf_matrix_percentage, 
            precision_class, recall_class, f1_class, 
            precision_avg, recall_avg, f1_avg)

def save_metrics_to_file(filename, class_names, precision_class, recall_class, f1_class, precision_avg, recall_avg, f1_avg):
    with open(filename, 'w') as f:
        f.write('Class-wise Precision, Recall, F1 Score:\n')
        for i, class_name in enumerate(class_names):
            f.write(f'{class_name}: Precision: {precision_class[i]:.4f}, Recall: {recall_class[i]:.4f}, F1 Score: {f1_class[i]:.4f}\n')
        
        f.write('\nAverage Precision, Recall, F1 Score:\n')
        f.write(f'Precision (avg): {precision_avg:.4f}, Recall (avg): {recall_avg:.4f}, F1 Score (avg): {f1_avg:.4f}\n')

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(root=args.test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

    model.load_state_dict(torch.load(args.model_path))

    criterion = torch.nn.CrossEntropyLoss()

    (test_loss, test_acc, conf_matrix, conf_matrix_percentage, 
     precision_class, recall_class, f1_class, 
     precision_avg, recall_avg, f1_avg) = test(model, test_loader, criterion, args.device, args.classes)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Confusion Matrix (Percentage):\n{conf_matrix_percentage}')
    print(f'Precision (Class-wise): {precision_class}')
    print(f'Recall (Class-wise): {recall_class}')
    print(f'F1 Score (Class-wise): {f1_class}')
    print(f'Precision (avg): {precision_avg:.4f}, Recall (avg): {recall_avg:.4f}, F1 Score (avg): {f1_avg:.4f}')

    save_path = args.result_save_path
    os.makedirs(save_path, exist_ok=True)
    plot_and_save_confusion_matrix(conf_matrix, args.classes, save_path + '/confusion_matrix.png', title='Confusion Matrix')
    plot_and_save_confusion_matrix(conf_matrix_percentage, args.classes, save_path + '/confusion_matrix_percentage.png', title='Confusion Matrix (Percentage)')

    save_metrics_to_file(save_path + '/test_metrics.txt', args.classes, precision_class, recall_class, f1_class, precision_avg, recall_avg, f1_avg)

def get_args():
    parser = argparse.ArgumentParser(description="Model configuration parser")

    parser.add_argument(
        "--model_name",
        type=str,
        default="OptimizedResnet101",
        help="Name of the model to be used. Valid models: OptimizedResnet101, OptimizedResnet50, Resnet101, Resnet50"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--test_data_path",
        type=str,
        default='',
        help="Path to the test dataset"
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
        "--device",
        type=str,
        default='cuda:1' if torch.cuda.is_available() else 'cpu',
        help="Device to use for training (cpu or cuda)"
    )

    parser.add_argument(
        "--result_save_path",
        type=str,
        default='',
        help="The path where the results will be saved"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default='',
        help="The path where the model is located"
    )

    return args

if __name__ == "__main__":
    args = get_args()
    main()

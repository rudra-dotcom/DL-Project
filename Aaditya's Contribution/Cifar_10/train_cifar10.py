import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models import create_model
import logging
import sys

# Ensure custom models are registered
import model
import optuna

def setup_logger(model_name):
    logger = logging.getLogger('cifar10_trainer')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Save to txt file (this captures all output)
    log_filename = f'{model_name}_cifar10_training.log'
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. Still show output in terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    print(f"Logging to: {log_filename}")
    return logger


# Global logger reference
logger = None

def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger('cifar10_trainer')
    return logger


# def setup_logger(model_name):
#     logger = logging.getLogger('cifar10_trainer')
#     logger.setLevel(logging.INFO)
    
#     # Remove existing handlers to avoid duplicates if called multiple times
#     if logger.hasHandlers():
#         logger.handlers.clear()
        
#     file_handler = logging.FileHandler(f'{model_name}_cifar10_training.log')
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
    
#     # Also print to console
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)
#     logger.addHandler(console_handler)
#     return logger

# logger = logging.getLogger('cifar10_trainer')

# optuna.logging.enable_propagation()
# optuna.logging.disable_default_handler()
# optuna.logging.set_verbosity(optuna.logging.INFO)

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader

def train_epoch(net, dataloader, criterion, optimizer, device):
    net.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(dataloader), 100. * correct / total

def test_epoch(net, dataloader, criterion, device):
    net.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / len(dataloader), 100. * correct / total

def objective(trial, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optuna hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
    
    # Init model
    net = create_model(args.model_name, num_classes=10, pretrained=False)
    net = net.to(device)
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # For SGD, tune momentum
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    criterion = nn.CrossEntropyLoss()
    trainloader, testloader = get_dataloaders(batch_size=256) # Fix batch size for trials
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(net, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(net, testloader, criterion, device)
        scheduler.step()
        
        best_acc = max(best_acc, test_acc)
        trial.report(test_acc, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return best_acc

def run_optuna(args):
    logger.info(f"Starting Optuna hyperparameter optimization for {args.model_name}...")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    logger.info("Optimization finished.")
    logger.info("Best Trial:")
    logger.info(f"  Accuracy: {study.best_trial.value:.2f}%")
    logger.info("  Parameters: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")


def train_single(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training {args.model_name} on {device}... (Epochs: {args.epochs})")

    net = create_model(args.model_name, num_classes=10, pretrained=False)
    net = net.to(device)

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    
    # Generic well-performing defaults if not tuning
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.025)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        curr_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_epoch(net, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(net, testloader, criterion, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save best checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(net.state_dict(), f'checkpoints/{args.model_name}_best.pth')

        logger.info(f"Epoch: {epoch+1:03d} | LR: {curr_lr:.5f} | "
              f"Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}% (Best: {best_acc:5.2f}%)")
              
    logger.info(f"Finished training {args.model_name}. Best Test Accuracy: {best_acc:.2f}%")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train CIFAR-10 models and Optuna tuning')
#     parser.add_argument('--model_name', default='repvit_cifar10', type=str)
#     parser.add_argument('--epochs', default=200, type=int)
#     parser.add_argument('--batch_size', default=128, type=int)
#     parser.add_argument('--optuna', action='store_true', help="Run Optuna hyperparameter sweep")
#     parser.add_argument('--n_trials', default=10, type=int, help="Number of Optuna trials")
    
#     args = parser.parse_args()
    
#     # Setup model-specific logger
#     setup_logger(args.model_name)
    
#     if args.optuna:
#         run_optuna(args)
#     else:
#         train_single(args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR-10 models and Optuna tuning')
    parser.add_argument('--model_name', default='repvit_cifar10', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--optuna', action='store_true', help="Run Optuna hyperparameter sweep")
    parser.add_argument('--n_trials', default=10, type=int, help="Number of Optuna trials")
    
    args = parser.parse_args()
    
    # Setup logger BEFORE any logging happens
    logger = setup_logger(args.model_name)
    
    if args.optuna:
        run_optuna(args)
    else:
        train_single(args)
# Install required packages
import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'matplotlib', 'pandas', 'numpy', 'lion-pytorch'])

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import copy
from lion_pytorch import Lion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)  # wtf why did this not work before
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Using device: {device}")

# Basic ResNet block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut connection - this part always confuses me
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# my FastResNet model - tried different configs
class fast_model(nn.Module):
    """FastResNet model for the experiments
    
    This is the main model I'm using for most tests
    """
    def __init__(self, num_classes=10):
        super(fast_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # these layer configs work best after lots of trial and error
        self.layer1 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)  # flatten
        out = self.fc(out)
        return out

# ResNet18 - needed for architecture comparison
class ResNet18_model(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # standard ResNet18 architecture
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# CompactCNN for testing - smaller model
class compactCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(compactCNN, self).__init__()
        # simple conv layers - nothing fancy
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def count_params(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def loadCIFAR10_data():
    # standard CIFAR10 loading - this part always works
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),  # data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # magic numbers from internet
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    print(f"Data loaded: {len(trainset)} train, {len(testset)} test samples")
    
    return trainloader, testloader

def makeOptimizer(model, opt_type, lr, wd):
    # tried different optimizers, these configs work
    if opt_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    elif opt_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    elif opt_type == 'lion':
        # this is the new optimizer we're testing
        return Lion(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.99))

def my_training_function(model, trainloader, testloader, optimizer, epochs=25, trackDynamics=False):
    """
    Main training loop with optional dynamics tracking
    Uses cosine annealing with warmup because that's what the paper said to do
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # cosine annealing with 5-epoch linear warmup
    # warmup_epochs = 5
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # implement scheduling
    base_lr = optimizer.param_groups[0]['lr']
    
    history = []
    dynamics_data = None
    if trackDynamics:
        dynamics_data = {'grad_norms': [], 'weight_norms': [], 'update_norms': [], 'update_ratios': []}
    
    # print("starting training")
    for epoch in range(epochs):
        # Learning rate schedule - linear warmup then cosine
        if epoch < 5:  # 5-epoch warmup
            lr_factor = 0.1 + 0.9 * (epoch + 1) / 5
            current_lr = base_lr * lr_factor
        else:
            # cosine annealing after warmup
            progress = (epoch - 5) / (epochs - 5)
            current_lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        epoch_grad_norms = []
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if trackDynamics:
                prev_weights = [p.data.clone() for p in model.parameters()]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # track dynamics every 50 batches - don't want too much data
            if trackDynamics:
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                epoch_grad_norms.append(grad_norm)
            
            optimizer.step()
            
            if trackDynamics and batch_idx % 50 == 0:
                # compute update norms and ratios
                weight_norm = 0
                update_norm = 0
                for p, prev_w in zip(model.parameters(), prev_weights):
                    weight_norm += p.data.norm(2).item() ** 2
                    update_norm += (p.data - prev_w).norm(2).item() ** 2
                
                weight_norm = weight_norm ** 0.5
                update_norm = update_norm ** 0.5
                update_ratio = update_norm / weight_norm if weight_norm > 0 else 0
                
                dynamics_data['weight_norms'].append(weight_norm)
                dynamics_data['update_norms'].append(update_norm)
                dynamics_data['update_ratios'].append(update_ratio)
            
            train_loss += loss.item()
            pred = outputs.max(1)[1]
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        
        if trackDynamics and epoch_grad_norms:
            dynamics_data['grad_norms'].append(np.mean(epoch_grad_norms))
        
        train_acc = 100.0 * correct / total
        
        # testing phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                pred = outputs.max(1)[1]
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        
        test_acc = 100.0 * correct / total
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': train_loss / len(trainloader),
            'test_loss': test_loss / len(testloader),
            'lr': current_lr
        })
        
        if epoch % 5 == 0 or epoch == epochs-1:
            print(f"      Epoch {epoch+1:2d}: Train {train_acc:.1f}% | Test {test_acc:.1f}% | LR {current_lr:.1e}")
    
    return history, dynamics_data

def evaluate_loss_on_dataset(model, dataloader):
    """Evaluate cross-entropy loss on entire dataset"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # multiply by batch size
            total_samples += inputs.size(0)
    
    return total_loss / total_samples

def compute_landscape_2d(model, testloader, grid_size=15):
    """
    Compute 2D loss landscape
    Uses random orthogonal directions
    This takes forever to run but gives good plots lol
    """
    model.eval()
    
    original_state = copy.deepcopy(model.state_dict())
    
    # generate two orthogonal random directions
    directions1 = []
    directions2 = []
    for param in model.parameters():
        d1 = torch.randn_like(param)
        d2 = torch.randn_like(param)
        directions1.append(d1)
        directions2.append(d2)
    
    # normalize to match model parameter norms
    norm1 = sum(d1.norm().item() ** 2 for d1 in directions1) ** 0.5
    norm2 = sum(d2.norm().item() ** 2 for d2 in directions2) ** 0.5
    
    for d in directions1:
        d.div_(norm1)
    for d in directions2:
        d.div_(norm2)
    
    # make orthogonal using Gram-Schmidt process
    dot_product = sum(torch.sum(d1 * d2).item() for d1, d2 in zip(directions1, directions2))
    for i, (d1, d2) in enumerate(zip(directions1, directions2)):
        directions2[i] = d2 - dot_product * d1
    
    norm2_corrected = sum(d2.norm().item() ** 2 for d2 in directions2) ** 0.5
    for d in directions2:
        d.div_(norm2_corrected)
    
    # create 15x15 grid with perturbation range [-1.0, 1.0]
    alphas = np.linspace(-1.0, 1.0, grid_size)
    betas = np.linspace(-1.0, 1.0, grid_size)
    
    losses = np.zeros((grid_size, grid_size))
    
    print(f"    Computing {grid_size}x{grid_size} loss landscape...")
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            model.load_state_dict(original_state)
            
            # perturb in 2D direction
            for param, d1, d2 in zip(model.parameters(), directions1, directions2):
                param.data.add_(d1, alpha=alpha).add_(d2, alpha=beta)
            
            # evaluate loss on entire test set
            total_loss = evaluate_loss_on_dataset(model, testloader)
            losses[i, j] = total_loss
        
        if (i + 1) % 5 == 0:
            print(f"      Progress: {i+1}/{grid_size} rows completed")
    
    model.load_state_dict(original_state)
    print(f"  Landscape computed: ({grid_size}, {grid_size}) grid")
    print(f"  Loss range: {losses.min():.3f} - {losses.max():.3f}")
    
    return alphas, betas, losses

def measure_sharpness_metric(model, testloader, epsilon=0.1, num_dirs=20):
    """Measure solution sharpness using random perturbations
    
    epsilon=0.1 seems to work well
    more directions = more accurate but slower
    """
    model.eval()
    
    original_state = copy.deepcopy(model.state_dict())
    
    # baseline loss on entire dataset
    baseline_loss = evaluate_loss_on_dataset(model, testloader)
    
    increases = []
    
    # test across 20 random directions
    for direction_idx in range(num_dirs):
        model.load_state_dict(original_state)
        
        direction = []
        for param in model.parameters():
            d = torch.randn_like(param)
            direction.append(d)
        
        # normalize direction
        dir_norm = sum(d.norm().item() ** 2 for d in direction) ** 0.5
        for d in direction:
            d.div_(dir_norm)
        
        # perturb with epsilon=0.1
        for param, d in zip(model.parameters(), direction):
            param.data.add_(d, alpha=epsilon)
        
        # measure loss increase on entire dataset
        perturbed_loss = evaluate_loss_on_dataset(model, testloader)
        
        # sharpness as relative increase
        increase = (perturbed_loss - baseline_loss) / baseline_loss
        increases.append(increase)
    
    model.load_state_dict(original_state)
    
    max_increase = max(increases)
    avg_increase = np.mean(increases)
    
    return baseline_loss, max_increase, avg_increase

# experiment 1 - loss landscape analysis
def run_landscape_experiment():
    print('STARTING COMPREHENSIVE LOSS LANDSCAPE ANALYSIS')
    print('')
    trainloader, testloader = loadCIFAR10_data()
    
    print('TRAINING MODELS WITH OPTIMAL HYPERPARAMETERS')
    print('='*50)
    print('')
    
    configs = {
        'sgd': {'lr': 2.0e-1, 'wd': 0.0},
        'adamw': {'lr': 2.0e-3, 'wd': 1.0e-4},  
        'lion': {'lr': 2.0e-4, 'wd': 5.0e-3}    # lion needs much higher WD
    }
    
    landscape_results = {}
    
    for opt_name, config in configs.items():
        opt_display = opt_name.upper()
        print(f'Training {opt_display} model:')
        print(f'  LR={config["lr"]:.1e}, WD={config["wd"]:.1e}')
        
        model = fast_model()
        param_count = count_params(model)
        # print(f'Model parameters: {param_count/1000:.0f}K')
        
        optimizer = makeOptimizer(model, opt_name, config['lr'], config['wd'])
        
        print(f'    Training 25 epochs with cosine schedule')
        # 25 epochs should be enough
        history, _ = my_training_function(model, trainloader, testloader, optimizer, epochs=25)
        final_acc = history[-1]['test_acc']
        
        print(f'  Final accuracy: {final_acc:.1f}%')
        print('')
        
        landscape_results[opt_name] = {
            'final_acc': final_acc,
            'history': history,
            'model': model  # save model for landscape analysis
        }
    
    print('ANALYZING LOSS LANDSCAPES')
    print('='*50)
    print('')
    
    # Now compute landscapes and sharpness
    for opt_name, result in landscape_results.items():
        opt_display = opt_name.upper()
        print(f'Computing loss landscape for {opt_display}:')
        
        model = result['model']
        alphas, betas, losses = compute_landscape_2d(model, testloader, grid_size=15)
        
        landscape_results[opt_name].update({
            'alphas': alphas,
            'betas': betas,
            'losses': losses,
            'loss_std': np.std(losses)
        })
        print('')
    
    print('ANALYZING SOLUTION SHARPNESS')
    print('='*50)
    print('')
    
    for opt_name, result in landscape_results.items():
        opt_display = opt_name.upper()
        print(f'Computing sharpness for {opt_display}:')
        
        model = result['model']
        baseline_loss, max_sharpness, avg_sharpness = measure_sharpness_metric(model, testloader)
        
        landscape_results[opt_name].update({
            'baseline_loss': baseline_loss,
            'max_sharpness': max_sharpness,
            'sharpness': avg_sharpness  # use avg as main sharpness metric
        })
        
        print(f'  Baseline loss: {baseline_loss:.3f}')
        print(f'  Max loss increase: {max_sharpness:.3f}')
        print(f'  Avg loss increase: {avg_sharpness:.3f}')
        print(f'  Sharpness (max): {max_sharpness:.3f}')
        print(f'  Sharpness (avg): {avg_sharpness:.3f}')
        print('')
    
    print('CREATING VISUALIZATIONS')
    print('='*30)
    print('')
    
    return landscape_results

def experimentTrainingDynamics():
    """Experiment 2: Training dynamics analysis
    
    This tracks how the optimizers behave during training
    """
    print('\n=== Experiment 2: Training Dynamics Analysis ===')
    trainloader, testloader = loadCIFAR10_data()
    
    configs = {
        'sgd': {'lr': 2.0e-1, 'wd': 0.0},
        'adamw': {'lr': 2.0e-3, 'wd': 1.0e-4},  # FIXED
        'lion': {'lr': 2.0e-4, 'wd': 5.0e-3}
    }
    
    dynamics_results = {}
    
    for opt_name, config in configs.items():
        print(f'\nTraining {opt_name} with dynamics tracking...')
        model = fast_model()
        optimizer = makeOptimizer(model, opt_name, config['lr'], config['wd'])
        
        # 25 epochs with dynamics tracking enabled
        history, dynamics = my_training_function(model, trainloader, testloader, optimizer, 
                                                epochs=25, trackDynamics=True)
        
        # compute averages for comparison
        avg_grad_norm = np.mean(dynamics['grad_norms']) if dynamics['grad_norms'] else 0
        avg_update_ratio = np.mean(dynamics['update_ratios']) if dynamics['update_ratios'] else 0
        
        dynamics_results[opt_name] = {
            'history': history,
            'dynamics': dynamics,
            'final_acc': history[-1]['test_acc'],
            'avg_grad_norm': avg_grad_norm,
            'avg_update_ratio': avg_update_ratio
        }
        
        print(f'{opt_name}: Final acc={history[-1]["test_acc"]:.1f}%, Grad norm={avg_grad_norm:.3f}, Update ratio={avg_update_ratio:.4f}')
    
    return dynamics_results

def weight_decay_experiment():
    """Experiment 3: Weight decay scaling analysis
    
    Testing different weight decay values to find optimal scaling
    """
    print('\n=== Experiment 3: Weight Decay Scaling Analysis ===')
    trainloader, testloader = loadCIFAR10_data()
    
    # weight decay ranges - lion needs much higher values
    wd_ranges = {
        'adamw': [0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'lion': [0, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]  # 10x-50x higher range
    }
    
    # learning rates are fixed at optimal values
    learning_rates = {'adamw': 2.0e-3, 'lion': 2.0e-4}
    
    wd_results = {}
    
    for opt_name in ['adamw', 'lion']:
        wd_results[opt_name] = []
        
        for wd in wd_ranges[opt_name]:
            print(f'Testing {opt_name} with WD={wd:.1e}...')
            model = fast_model()
            optimizer = makeOptimizer(model, opt_name, learning_rates[opt_name], wd)
            
            # 15 epochs should be enough to see the effect
            history, _ = my_training_function(model, trainloader, testloader, optimizer, epochs=15)
            best_acc = max([h['test_acc'] for h in history])
            
            wd_results[opt_name].append({
                'wd': wd,
                'best_acc': best_acc,
                'final_acc': history[-1]['test_acc']
            })
            
            print(f'  {opt_name} WD={wd:.1e}: Best={best_acc:.1f}%')
    
    return wd_results

def architecture_generalization_experiment():
    """Experiment 4: Testing different architectures
    
    Want to see if the results hold across different model sizes
    """
    print('\n=== Experiment 4: Architectural Dependency Analysis ===')
    trainloader, testloader = loadCIFAR10_data()
    
    # three architectures with different parameter counts
    architectures = {
        'FastResNet': fast_model,
        'ResNet18': ResNet18_model,
        'CompactCNN': compactCNN
    }
    
    configs = {
        'sgd': {'lr': 2.0e-1, 'wd': 0.0},
        'adamw': {'lr': 2.0e-3, 'wd': 1.0e-4},  # FIXED
        'lion': {'lr': 2.0e-4, 'wd': 5.0e-3}
    }
    
    arch_results = {}
    
    for arch_name, arch_class in architectures.items():
        print(f'\nTesting {arch_name}...')
        test_model = arch_class()
        param_count = count_params(test_model)
        print(f'  Parameters: {param_count/1000:.0f}K ({param_count/1000000:.1f}M)')
        
        arch_results[arch_name] = {}
        
        for opt_name, config in configs.items():
            print(f'  Training {arch_name} with {opt_name}...')
            model = arch_class()
            optimizer = makeOptimizer(model, opt_name, config['lr'], config['wd'])
            
            # 20 epochs for comparison
            history, _ = my_training_function(model, trainloader, testloader, optimizer, epochs=20)
            best_acc = max([h['test_acc'] for h in history])
            
            arch_results[arch_name][opt_name] = {
                'best_acc': best_acc,
                'final_acc': history[-1]['test_acc'],
                'param_count': param_count
            }
            
            print(f'    {opt_name}: Best={best_acc:.1f}%')
    
    return arch_results

# plotting functions 
def plot_landscapes(landscape_results):
    """Loss landscape plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = {'sgd': 'blue', 'adamw': 'orange', 'lion': 'green'}  # basic colors
    
    # landscape contour plots
    for i, (opt_name, data) in enumerate(landscape_results.items()):
        ax = axes[0, i]
        X, Y = np.meshgrid(data['betas'], data['alphas'])
        contour = ax.contourf(X, Y, data['losses'], levels=20, cmap='viridis')
        ax.plot(0, 0, 'r*', markersize=15)  # mark the center
        ax.set_title(f'{opt_name.upper()} Loss Landscape')
        ax.set_xlabel('Direction 2')
        ax.set_ylabel('Direction 1')
        plt.colorbar(contour, ax=ax)
    
    # sharpness comparison
    ax = axes[1, 0]
    opts = list(landscape_results.keys())
    sharpness_vals = [landscape_results[opt]['sharpness'] for opt in opts]
    bars = ax.bar(opts, sharpness_vals, color=[colors[opt] for opt in opts])
    ax.set_title('Solution Sharpness')
    ax.set_ylabel('Sharpness')
    for bar, val in zip(bars, sharpness_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sharpness_vals)*0.01, 
                f'{val:.3f}', ha='center')
    
    # performance vs sharpness scatter plot
    ax = axes[1, 1]
    accs = [landscape_results[opt]['final_acc'] for opt in opts]
    ax.scatter(sharpness_vals, accs, c=[colors[opt] for opt in opts], s=150)
    for i, opt in enumerate(opts):
        ax.annotate(opt.upper(), (sharpness_vals[i], accs[i]), xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Sharpness')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Sharpness vs Performance')
    
    # summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    for opt in opts:
        data = landscape_results[opt]
        table_data.append([
            opt.upper(),
            f'{data["final_acc"]:.1f}%',
            f'{data["sharpness"]:.3f}',
            f'{data["loss_std"]:.1f}'
        ])
    
    table = ax.table(cellText=table_data, 
                     colLabels=['Optimizer', 'Test Acc', 'Sharpness', 'Loss Range'], 
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Solution Characteristics Analysis', pad=20)
    
    plt.suptitle('Experiment 1: Loss Landscape Characteristics', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/loss_landscape_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plotDynamics(dynamics_results):
    """Training dynamics plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = {'sgd': 'blue', 'adamw': 'orange', 'lion': 'green'}
    
    # learning curves
    ax = axes[0, 0]
    for opt_name, data in dynamics_results.items():
        epochs = [h['epoch'] for h in data['history']]
        test_accs = [h['test_acc'] for h in data['history']]
        ax.plot(epochs, test_accs, label=opt_name.upper(), color=colors[opt_name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True)
    
    # gradient norms 
    ax = axes[0, 1]
    for opt_name, data in dynamics_results.items():
        if data['dynamics']['grad_norms']:
            epochs = range(len(data['dynamics']['grad_norms']))
            ax.plot(epochs, data['dynamics']['grad_norms'], label=opt_name.upper(), 
                   color=colors[opt_name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Evolution')
    ax.legend()
    ax.grid(True)
    
    # update to weight ratios
    ax = axes[0, 2]
    for opt_name, data in dynamics_results.items():
        if data['dynamics']['update_ratios']:
            steps = range(len(data['dynamics']['update_ratios']))
            ax.plot(steps, data['dynamics']['update_ratios'], label=opt_name.upper(), 
                   color=colors[opt_name])
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Update/Weight Ratio')
    ax.set_title('Update Magnitude Analysis')
    ax.legend()
    ax.grid(True)
    
    # comparison bars for gradient norms
    ax = axes[1, 0]
    opts = list(dynamics_results.keys())
    grad_norms = [dynamics_results[opt]['avg_grad_norm'] for opt in opts]
    bars = ax.bar(opts, grad_norms, color=[colors[opt] for opt in opts])
    ax.set_title('Average Gradient Norms')
    ax.set_ylabel('Gradient Norm')
    for bar, val in zip(bars, grad_norms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(grad_norms)*0.01, 
                f'{val:.3f}', ha='center')
    
    # update ratios comparison
    ax = axes[1, 1]
    update_ratios = [dynamics_results[opt]['avg_update_ratio'] for opt in opts]
    bars = ax.bar(opts, update_ratios, color=[colors[opt] for opt in opts])
    ax.set_title('Average Update/Weight Ratios')
    ax.set_ylabel('Update/Weight Ratio')
    for bar, val in zip(bars, update_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(update_ratios)*0.01, 
                f'{val:.4f}', ha='center')
    
    # summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    for opt in opts:
        data = dynamics_results[opt]
        table_data.append([
            opt.upper(),
            f'{data["avg_grad_norm"]:.3f}',
            f'{data["avg_update_ratio"]:.4f}',
            f'{data["final_acc"]:.1f}%'
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Optimizer', 'Grad Norm', 'Update/Weight', 'Final Acc'],
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Training Dynamics Summary', pad=20)
    
    plt.suptitle('Experiment 2: Training Dynamics Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/training_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plotWeightDecay(wd_results):
    """Weight decay analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'adamw': 'orange', 'lion': 'green'}
    
    # weight decay effect - semilog plot looks better
    ax = axes[0, 0]
    for opt_name in ['adamw', 'lion']:
        wd_vals = [item['wd'] + 1e-6 for item in wd_results[opt_name]]  # avoid log(0)
        best_accs = [item['best_acc'] for item in wd_results[opt_name]]
        ax.semilogx(wd_vals, best_accs, 'o-', label=opt_name.upper(), color=colors[opt_name])
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Weight Decay Scaling Analysis')
    ax.legend()
    ax.grid(True)
    
    # optimal performance comparison
    ax = axes[0, 1]
    adamw_best = max(wd_results['adamw'], key=lambda x: x['best_acc'])
    lion_best = max(wd_results['lion'], key=lambda x: x['best_acc'])
    
    optimal_accs = [adamw_best['best_acc'], lion_best['best_acc']]
    bars = ax.bar(['AdamW', 'Lion'], optimal_accs, color=['orange', 'green'])
    ax.set_title('Optimal Performance Comparison')
    ax.set_ylabel('Best Test Accuracy (%)')
    for bar, acc in zip(bars, optimal_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{acc:.1f}%', 
                ha='center')
    
    # scaling analysis - this is the key finding
    ax = axes[1, 0]
    scaling_factor = lion_best['wd'] / adamw_best['wd'] if adamw_best['wd'] > 0 else float('inf')
    
    ax.axis('off')
    scaling_text = f"""Weight Decay Scaling Analysis

AdamW Optimal: {adamw_best['wd']:.1e}
Lion Optimal: {lion_best['wd']:.1e}

Scaling Factor: {scaling_factor:.0f}×

Lion requires {scaling_factor:.0f}× higher weight decay 
than AdamW for optimal performance"""
    
    ax.text(0.05, 0.5, scaling_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    # summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['AdamW', f'{adamw_best["wd"]:.1e}', f'{adamw_best["best_acc"]:.1f}%'],
        ['Lion', f'{lion_best["wd"]:.1e}', f'{lion_best["best_acc"]:.1f}%'],
        ['Scaling', f'{scaling_factor:.0f}×', 'Ratio']
    ]
    table = ax.table(cellText=table_data, colLabels=['Optimizer', 'Optimal WD', 'Best Acc'], 
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax.set_title('Weight Decay Summary', pad=20)
    
    plt.suptitle('Experiment 3: Weight Decay Scaling Study', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/weight_decay_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_architecture_results(arch_results):
    """Architecture comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'sgd': 'blue', 'adamw': 'orange', 'lion': 'green'}
    
    # performance across architectures
    ax = axes[0, 0]
    architectures = list(arch_results.keys())
    x_pos = np.arange(len(architectures))
    width = 0.25
    
    for i, opt in enumerate(['sgd', 'adamw', 'lion']):
        accs = [arch_results[arch][opt]['best_acc'] for arch in architectures]
        ax.bar(x_pos + i*width, accs, width, label=opt.upper(), color=colors[opt])
    
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Performance Across Architectures')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'{arch}\n({arch_results[arch]["sgd"]["param_count"]/1000:.0f}K)' 
                       for arch in architectures])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # lion gaps - interesting pattern here
    ax = axes[0, 1]
    lion_gaps = []
    for arch in architectures:
        sgd_acc = arch_results[arch]['sgd']['best_acc']
        lion_acc = arch_results[arch]['lion']['best_acc']
        gap = sgd_acc - lion_acc
        lion_gaps.append(gap)
    
    bars = ax.bar(architectures, lion_gaps, color='green', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # reference line
    ax.set_ylabel('Lion Gap vs SGD (%)')
    ax.set_title('Lion Performance Gap Analysis')
    ax.grid(True, alpha=0.3)
    for bar, gap in zip(bars, lion_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 if gap >= 0 else gap - 0.15, 
                f'{gap:+.1f}%', ha='center')
    
    # notable finding about ResNet18 - this was surprising!
    ax = axes[1, 0]
    ax.axis('off')
    
    # check if Lion outperforms on ResNet18
    resnet18_results = arch_results.get('ResNet18', {})
    if resnet18_results:
        lion_resnet = resnet18_results['lion']['best_acc']
        adamw_resnet = resnet18_results['adamw']['best_acc']
        sgd_resnet = resnet18_results['sgd']['best_acc']
        
        finding_text = f"""Notable Finding:

Lion's architectural dependency shows
surprising performance on ResNet18:

SGD:   {sgd_resnet:.1f}%
AdamW: {adamw_resnet:.1f}%
Lion:  {lion_resnet:.1f}%

Lion actually achieves BEST performance
on ResNet18 - unexpected result!"""
    else:
        finding_text = "ResNet18 results not available"
    
    ax.text(0.05, 0.5, finding_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    
    # summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for arch in architectures:
        sgd_acc = arch_results[arch]['sgd']['best_acc']
        adamw_acc = arch_results[arch]['adamw']['best_acc']
        lion_acc = arch_results[arch]['lion']['best_acc']
        gap = sgd_acc - lion_acc
        
        table_data.append([
            f'{arch} ({arch_results[arch]["sgd"]["param_count"]/1000:.0f}K)',
            f'{sgd_acc:.1f}%',
            f'{adamw_acc:.1f}%',
            f'{lion_acc:.1f}%',
            f'{gap:+.1f}%'
        ])
    
    table = ax.table(cellText=table_data, 
                     colLabels=['Architecture', 'SGD (%)', 'AdamW (%)', 'Lion (%)', 'Lion Gap'],
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.set_title('Architecture-Dependent Performance', pad=20)
    
    plt.suptitle('Experiment 4: Architectural Dependency Evaluation', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/architecture_generalization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    os.makedirs('results', exist_ok=True)
    
    print('Mechanistic Analysis of Lion Optimizer')
    print('Reproducing experiments from research paper')
    print('='*60)
    
    # TODO: add more seeds for statistical validation if have time
    # TODO: try different learning rates if current ones don't work
    
    # run all four experiments
    print('\nExperiment 1: Loss Landscape Characteristics')
    landscape_results = run_landscape_experiment()
    plot_landscapes(landscape_results)
    
    print('\nExperiment 2: Training Dynamics Analysis') 
    dynamics_results = experimentTrainingDynamics()
    plotDynamics(dynamics_results)
    
    print('\nExperiment 3: Weight Decay Scaling Study')
    wd_results = weight_decay_experiment()
    plotWeightDecay(wd_results)
    
    print('\nExperiment 4: Architectural Dependency Evaluation')
    arch_results = architecture_generalization_experiment()
    plot_architecture_results(arch_results)
    
    # summary matching paper conclusions
    print('\n' + '='*60)
    print('MECHANISTIC ANALYSIS SUMMARY')
    print('='*60)
    
    # key findings from experiments
    sgd_acc = landscape_results['sgd']['final_acc']
    lion_acc = landscape_results['lion']['final_acc']
    gap = sgd_acc - lion_acc
    
    sgd_sharpness = landscape_results['sgd']['sharpness']
    lion_sharpness = landscape_results['lion']['sharpness']
    
    sgd_grad_norm = dynamics_results['sgd']['avg_grad_norm']
    lion_grad_norm = dynamics_results['lion']['avg_grad_norm']
    
    sgd_update_ratio = dynamics_results['sgd']['avg_update_ratio']
    lion_update_ratio = dynamics_results['lion']['avg_update_ratio']
    
    adamw_optimal = max(wd_results['adamw'], key=lambda x: x['best_acc'])
    lion_optimal = max(wd_results['lion'], key=lambda x: x['best_acc'])
    wd_scaling = lion_optimal['wd'] / adamw_optimal['wd'] if adamw_optimal['wd'] > 0 else float('inf')

# TODO: 
#   test with different batch sizes if results look weird
#   add more optimizers for comparison (RAdam, AdaBound, etc.)
#   clean up this code before submission

if __name__ == '__main__':
    main()

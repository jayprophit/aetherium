"""
Image Classification Example

This example demonstrates how to use the AI systems components to train and evaluate
a simple CNN on the CIFAR-10 dataset.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import AI systems components
from ai.inference import Predictor, InferencePipeline
from ai.inference.postprocessing import ClassificationPostProcessor
from ai.evaluation import ModelEvaluator
from ai.evaluation.visualization import plot_confusion_matrix, plot_training_history
from ai.models.cnn import SimpleCNN, ResNet18  # Import new model implementations
from ai.training.trainer import Trainer  # Import trainer

# Set random seed for reproducibility
torch.manual_seed(42)

# Configuration
config = {
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'num_classes': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './data/cifar10',
    'output_dir': './output'
}

# Create output directory
os.makedirs(config['output_dir'], exist_ok=True)

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
def load_datasets():
    """Load and prepare CIFAR-10 datasets."""
    # Download and load training data
    full_train = torchvision.datasets.CIFAR10(
        root=config['data_dir'],
        train=True,
        download=True,
        transform=transform
    )
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    
    # Load test set
    test_set = torchvision.datasets.CIFAR10(
        root=config['data_dir'],
        train=False,
        download=True,
        transform=transform
    )
    
    return train_set, val_set, test_set

def main():
    # Load datasets
    train_set, val_set, test_set = load_datasets()
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = ResNet18(num_classes=config['num_classes']).to(config['device'])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        device=config['device'],
        log_dir=os.path.join(config['output_dir'], 'logs'),
        checkpoint_dir=os.path.join(config['output_dir'], 'checkpoints'),
        save_every=5,
        log_every=100
    )
    
    # Training loop
    print("Starting training...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(config['num_epochs']):
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        
        # Record metrics
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}% | "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(config['output_dir'], 'model.pth'))
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(config['output_dir'], 'training_history.png')
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.2f}%")
    
    # Generate confusion matrix
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config['device']), labels.to(config['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        save_path=os.path.join(config['output_dir'], 'confusion_matrix.png')
    )
    
    # Set up inference pipeline
    predictor = Predictor(model, device=config['device'])
    postprocessor = ClassificationPostProcessor(top_k=3)
    
    pipeline = InferencePipeline(steps=[
        ('predict', predictor),
        ('postprocess', postprocessor)
    ])
    
    # Example inference
    print("\nRunning example inference...")
    image, label = test_set[0]
    image = image.unsqueeze(0).to(config['device'])  # Add batch dimension
    
    result = pipeline.run(image)
    print(f"Predicted: {class_names[result['postprocess']['top_classes'][0]]} "
          f"(True: {class_names[label]})")
    print(f"Top predictions: {[class_names[i] for i in result['postprocess']['top_classes']]}")
    print(f"Probabilities: {result['postprocess']['probabilities']}")

if __name__ == '__main__':
    main()

import argparse
import os
from preprocessing import HieroglyphDataProcessor
from model import HieroglyphModel
import tensorflow as tf
import json

def main(args):
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data processor
    processor = HieroglyphDataProcessor(
        dataset_path=args.dataset_path,
        output_path=args.output_dir,
        img_size=(args.img_width, args.img_height)
    )
    
    # Load and prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, num_classes = processor.process_and_prepare_datasets(
        batch_size=args.batch_size
    )
    
    # Load the class mapping
    with open(os.path.join(args.output_dir, 'class_mapping.json'), 'r') as f:
        class_mapping = json.load(f)
    
    # Print dataset information
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping saved to {os.path.join(args.output_dir, 'class_mapping.json')}")
    
    # Initialize model
    model_instance = HieroglyphModel(
        num_classes=num_classes,
        img_size=(args.img_width, args.img_height),
        model_dir=os.path.join(args.output_dir, 'model')
    )
    
    # Build model based on selected architecture
    print(f"Building {args.model_type} model...")
    if args.model_type == 'cnn':
        model = model_instance.build_cnn_model()
    elif args.model_type == 'efficientnet':
        model = model_instance.build_efficientnet_model(freeze_base=not args.unfreeze_base)
    elif args.model_type == 'transformer':
        model = model_instance.build_transformer_model()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Model summary
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model_instance.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    # Plot training history
    model_instance.plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = model_instance.evaluate(test_dataset)
    
    # Load test split dataframe for visualization
    import pandas as pd
    test_df = pd.read_csv(os.path.join(args.output_dir, 'test_split.csv'))
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    model_instance.visualize_predictions(test_df, processor, num_samples=20)
    
    # Export model for deployment
    if args.export_model:
        print("\nExporting model for deployment...")
        model_instance.export_for_deployment(
            export_dir=os.path.join(args.output_dir, 'deployment')
        )
    
    print(f"\nTraining complete! All outputs saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hieroglyph recognition model")
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='Dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='hieroglyph_recognition',
                        help='Directory to save processed data, models, and results')
    
    # Image parameters
    parser.add_argument('--img_height', type=int, default=50,
                        help='Height of input images')
    parser.add_argument('--img_width', type=int, default=75,
                        help='Width of input images')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs to train')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'efficientnet', 'transformer'],
                        help='Type of model to train')
    parser.add_argument('--unfreeze_base', action='store_true',
                        help='Unfreeze base model for fine-tuning (only for efficientnet)')
    
    # Export options
    parser.add_argument('--export_model', action='store_true',
                        help='Export model for deployment')
    
    args = parser.parse_args()
    main(args) 
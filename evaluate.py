import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from preprocessing import HieroglyphDataProcessor
from model import HieroglyphModel

def load_model_and_metadata(model_path, output_dir):
    """Load trained model, class mapping, and test data"""
    # Load model
    model_path = os.path.join(model_path, 'final_model.h5')
    model = tf.keras.models.load_model(model_path)
    
    # Load class mapping
    with open(os.path.join(output_dir, 'class_mapping.json'), 'r') as f:
        class_mapping_data = json.load(f)
        
    class_mapping = class_mapping_data['class_mapping']
    inv_class_mapping = class_mapping_data['inv_class_mapping']
    
    # Convert string keys in inv_class_mapping to integers
    inv_class_mapping = {int(k): v for k, v in inv_class_mapping.items()}
    
    # Load test split
    test_df = pd.read_csv(os.path.join(output_dir, 'test_split.csv'))
    
    return model, class_mapping, inv_class_mapping, test_df

def predict_test_set(model, test_dataset, test_df, inv_class_mapping):
    """Make predictions on the test set"""
    # Get ground truth labels
    y_true = test_df['class_idx'].values
    
    # Get predictions
    predictions = []
    prediction_classes = []
    
    for images, labels in test_dataset:
        batch_predictions = model.predict(images)
        predictions.extend(batch_predictions)
        prediction_classes.extend(np.argmax(batch_predictions, axis=1))
    
    # Convert to numpy arrays
    y_pred = np.array(prediction_classes)
    
    return y_true, y_pred, predictions

def visualize_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Create and save confusion matrix visualization"""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Save top misclassifications
    misclassified = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            misclassified.append({
                'true_class': class_names[y_true[i]],
                'predicted_class': class_names[y_pred[i]]
            })
    
    misclass_df = pd.DataFrame(misclassified)
    misclass_counts = misclass_df.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
    misclass_counts = misclass_counts.sort_values('count', ascending=False).head(20)
    
    # Save top misclassifications
    misclass_counts.to_csv(os.path.join(output_dir, 'top_misclassifications.csv'), index=False)
    
    # Visualize top misclassifications
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='true_class', hue='predicted_class', data=misclass_counts.head(15))
    plt.title('Top 15 Misclassifications')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_misclassifications.png'))
    plt.close()

def visualize_challenging_examples(test_df, processor, model, inv_class_mapping, output_dir, num_samples=20):
    """Visualize examples with the lowest prediction confidence"""
    # Create HieroglyphModel instance
    model_instance = HieroglyphModel(
        num_classes=len(inv_class_mapping),
        img_size=processor.img_size
    )
    model_instance.model = model
    
    # Make predictions on all test images
    all_predictions = []
    for _, row in test_df.iterrows():
        pred = model_instance.predict_hieroglyph(row['path'], processor)
        pred['true_class'] = row['gardiner_code']
        pred['img_path'] = row['path']
        all_predictions.append(pred)
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(all_predictions)
    
    # Find challenging examples (lowest confidence predictions)
    challenging = pred_df.sort_values('confidence').head(num_samples)
    
    # Visualize
    rows = (num_samples + 4) // 5
    cols = min(5, num_samples)
    
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for i, (_, row) in enumerate(challenging.iterrows()):
        # Get image
        img_path = row['img_path']
        true_label = row['true_class']
        pred_label = row['gardiner_code']
        confidence = row['confidence']
        
        # Display image with prediction
        img = processor.preprocess_image(img_path) * 255.0
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.astype(np.uint8))
        
        # Set title color based on prediction (green for correct, red for wrong)
        title_color = 'green' if pred_label == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})", 
                  color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'challenging_examples.png'))
    plt.close()
    
    # Save predictions for analysis
    pred_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)

def compute_and_save_metrics(y_true, y_pred, class_names, output_dir):
    """Compute and save classification metrics"""
    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Save to CSV
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Print summary
    print("\nClassification Report Summary:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    # Save per-class F1 scores
    class_metrics = []
    for i, class_name in enumerate(class_names):
        if class_name in report:
            class_metrics.append({
                'class': class_name,
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1-score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            })
    
    class_metrics_df = pd.DataFrame(class_metrics)
    
    # Sort by F1-score to see best and worst performing classes
    class_metrics_df = class_metrics_df.sort_values('f1-score')
    
    # Save to CSV
    class_metrics_df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)
    
    # Visualize worst and best performing classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Worst performing
    worst_df = class_metrics_df.head(10)
    sns.barplot(x='f1-score', y='class', data=worst_df, ax=ax1)
    ax1.set_title('10 Worst Performing Classes (F1-Score)')
    ax1.set_xlim(0, 1)
    
    # Best performing
    best_df = class_metrics_df.tail(10).iloc[::-1]  # Reverse to show best at top
    sns.barplot(x='f1-score', y='class', data=best_df, ax=ax2)
    ax2.set_title('10 Best Performing Classes (F1-Score)')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'))
    plt.close()

def main(args):
    # Setup output directory for evaluation results
    eval_output_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Load model and metadata
    print("Loading model and metadata...")
    model, class_mapping, inv_class_mapping, test_df = load_model_and_metadata(
        model_path=os.path.join(args.output_dir, 'model'),
        output_dir=args.output_dir
    )
    
    # Initialize data processor
    processor = HieroglyphDataProcessor(
        dataset_path=args.dataset_path,
        output_path=args.output_dir,
        img_size=(args.img_width, args.img_height)
    )
    
    # Re-create test dataset
    print("Creating test dataset...")
    test_dataset = processor.create_tf_dataset(
        test_df, subset='test', batch_size=args.batch_size, shuffle=False, augment=False
    )
    
    # Make predictions on test set
    print("Predicting on test set...")
    y_true, y_pred, predictions = predict_test_set(model, test_dataset, test_df, inv_class_mapping)
    
    # Get class names
    class_names = [inv_class_mapping[i] for i in range(len(inv_class_mapping))]
    
    # Compute and save metrics
    print("Computing metrics...")
    compute_and_save_metrics(y_true, y_pred, class_names, eval_output_dir)
    
    # Visualize confusion matrix
    print("Generating confusion matrix...")
    visualize_confusion_matrix(y_true, y_pred, class_names, eval_output_dir)
    
    # Visualize challenging examples
    print("Visualizing challenging examples...")
    visualize_challenging_examples(test_df, processor, model, inv_class_mapping, eval_output_dir)
    
    print(f"\nEvaluation complete! Results saved to {eval_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained hieroglyph recognition model")
    
    # Input parameters
    parser.add_argument('--output_dir', type=str, default='hieroglyph_recognition',
                        help='Directory containing the trained model and processed data')
    parser.add_argument('--dataset_path', type=str, default='Dataset',
                        help='Path to the original dataset directory')
    
    # Image parameters
    parser.add_argument('--img_height', type=int, default=50,
                        help='Height of input images')
    parser.add_argument('--img_width', type=int, default=75,
                        help='Width of input images')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    main(args) 
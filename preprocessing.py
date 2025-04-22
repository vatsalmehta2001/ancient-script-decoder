import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import tensorflow as tf
import pandas as pd
from PIL import Image
import albumentations as A

class HieroglyphDataProcessor:
    def __init__(self, dataset_path='Dataset', output_path='processed_data', img_size=(75, 50)):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.img_size = img_size
        
        # Create output directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_path / 'train', exist_ok=True)
        os.makedirs(self.output_path / 'val', exist_ok=True)
        os.makedirs(self.output_path / 'test', exist_ok=True)
        
        # Map Gardiner classes to integers
        self.class_mapping = {}
        self.inv_class_mapping = {}
        
    def extract_gardiner_code(self, filename):
        """Extract Gardiner code from filename"""
        match = re.search(r'_([A-Z][0-9]+|UNKNOWN)\.png$', filename)
        if match:
            return match.group(1)
        return 'UNKNOWN'
    
    def load_and_organize_data(self, subset='Manual'):
        """Load and organize hieroglyph data from the dataset"""
        data = []
        
        # Walk through all preprocessed directories
        preprocessed_dir = self.dataset_path / subset / 'Preprocessed'
        for picture_dir in os.listdir(preprocessed_dir):
            if not os.path.isdir(preprocessed_dir / picture_dir):
                continue
                
            for img_file in os.listdir(preprocessed_dir / picture_dir):
                img_path = preprocessed_dir / picture_dir / img_file
                gardiner_code = self.extract_gardiner_code(img_file)
                
                # Skip unknown class if needed
                if gardiner_code == 'UNKNOWN':
                    continue
                    
                data.append({
                    'path': str(img_path),
                    'gardiner_code': gardiner_code,
                    'source_image': picture_dir
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create class mapping
        unique_classes = sorted(df['gardiner_code'].unique())
        self.class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        self.inv_class_mapping = {i: cls for i, cls in enumerate(unique_classes)}
        
        # Add class index to dataframe
        df['class_idx'] = df['gardiner_code'].map(self.class_mapping)
        
        return df
    
    def create_data_splits(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """Split data into train, validation, and test sets"""
        # Filter out classes with only one sample
        class_counts = df['gardiner_code'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        df_filtered = df[df['gardiner_code'].isin(valid_classes)].copy()
        
        if len(df_filtered) < len(df):
            removed_count = len(df) - len(df_filtered)
            print(f"Removed {removed_count} samples from {len(class_counts) - len(valid_classes)} classes with only one sample")
            
            # Update class mappings
            unique_classes = sorted(df_filtered['gardiner_code'].unique())
            self.class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
            self.inv_class_mapping = {i: cls for i, cls in enumerate(unique_classes)}
            
            # Update class index in dataframe
            df_filtered['class_idx'] = df_filtered['gardiner_code'].map(self.class_mapping)
        
        # First split: training + validation vs test
        train_val_df, test_df = train_test_split(
            df_filtered, test_size=test_size, random_state=random_state, stratify=df_filtered['gardiner_code']
        )
        
        # Second split: training vs validation
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size/(1-test_size),
            random_state=random_state,
            stratify=train_val_df['gardiner_code']
        )
        
        print(f"Training set: {len(train_df)} images")
        print(f"Validation set: {len(val_df)} images")
        print(f"Test set: {len(test_df)} images")
        
        return train_df, val_df, test_df
    
    def get_augmentation_pipeline(self):
        """Define augmentation pipeline for training data"""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            ], p=0.3),
        ])
    
    def preprocess_image(self, img_path, augment=False):
        """Load and preprocess a single image"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations for training data
        if augment:
            augmentation = self.get_augmentation_pipeline()
            transformed = augmentation(image=img)
            img = transformed['image']
        
        # Resize to target size
        img = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def create_tf_dataset(self, df, subset='train', batch_size=32, shuffle=True, augment=False):
        """Create TensorFlow dataset from DataFrame"""
        def generator():
            for _, row in df.iterrows():
                img = self.preprocess_image(row['path'], augment=augment)
                yield img, row['class_idx']
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.img_size[1], self.img_size[0], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def save_class_mapping(self):
        """Save class mapping to JSON file"""
        import json
        with open(self.output_path / 'class_mapping.json', 'w') as f:
            json.dump({
                'class_mapping': self.class_mapping,
                'inv_class_mapping': self.inv_class_mapping
            }, f, indent=2)
    
    def process_and_prepare_datasets(self, batch_size=32):
        """Process data and prepare TensorFlow datasets"""
        # Load and organize data
        df = self.load_and_organize_data(subset='Manual')
        
        # Split data
        train_df, val_df, test_df = self.create_data_splits(df)
        
        # Save splits for reference
        train_df.to_csv(self.output_path / 'train_split.csv', index=False)
        val_df.to_csv(self.output_path / 'val_split.csv', index=False)
        test_df.to_csv(self.output_path / 'test_split.csv', index=False)
        
        # Create TensorFlow datasets
        train_dataset = self.create_tf_dataset(train_df, subset='train', batch_size=batch_size, augment=True)
        val_dataset = self.create_tf_dataset(val_df, subset='val', batch_size=batch_size, augment=False)
        test_dataset = self.create_tf_dataset(test_df, subset='test', batch_size=batch_size, augment=False)
        
        # Save class mapping
        self.save_class_mapping()
        
        return train_dataset, val_dataset, test_dataset, len(self.class_mapping)
    
    def visualize_samples(self, df, num_samples=5, cols=5):
        """Visualize random samples from the dataset"""
        samples = df.sample(num_samples)
        rows = (num_samples + cols - 1) // cols
        
        plt.figure(figsize=(cols * 3, rows * 3))
        for i, (_, row) in enumerate(samples.iterrows()):
            img = self.preprocess_image(row['path']) * 255.0
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img.astype(np.uint8))
            plt.title(f"{row['gardiner_code']}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'sample_hieroglyphs.png')
        plt.close()
    
    def visualize_augmentations(self, img_path, num_samples=5):
        """Visualize augmentations for a single image"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        augmentation = self.get_augmentation_pipeline()
        
        plt.figure(figsize=(15, 3))
        plt.subplot(1, num_samples + 1, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.axis('off')
        
        for i in range(num_samples):
            augmented = augmentation(image=img.copy())['image']
            plt.subplot(1, num_samples + 1, i + 2)
            plt.imshow(augmented)
            plt.title(f'Augmented {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'augmentation_examples.png')
        plt.close()

if __name__ == "__main__":
    processor = HieroglyphDataProcessor()
    
    # Load and organize data
    df = processor.load_and_organize_data()
    
    # Split data
    train_df, val_df, test_df = processor.create_data_splits(df)
    
    # Visualize samples
    processor.visualize_samples(df, num_samples=10)
    
    # Visualize augmentations for a random image
    random_img = df.sample(1).iloc[0]['path']
    processor.visualize_augmentations(random_img)
    
    # Process and prepare datasets
    train_dataset, val_dataset, test_dataset, num_classes = processor.process_and_prepare_datasets()
    
    print(f"Dataset preparation complete. Number of classes: {num_classes}") 
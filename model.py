import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from pathlib import Path

class HieroglyphModel:
    def __init__(self, num_classes, img_size=(75, 50), model_dir='models'):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_dir = Path(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model = None
        
    def build_cnn_model(self):
        """Build and compile a CNN model for hieroglyph classification"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.img_size[1], self.img_size[0], 3)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def build_efficientnet_model(self, freeze_base=True):
        """Build a transfer learning model using EfficientNetB0"""
        # Use EfficientNetB0 as base model
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(self.img_size[1], self.img_size[0], 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model
        if freeze_base:
            base_model.trainable = False
        
        # Add classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
        
    def build_transformer_model(self):
        """Build a Vision Transformer (ViT) inspired model for hieroglyph classification"""
        
        # Patch size and projection dimension
        patch_size = 5  # We'll use 5x5 patches
        projection_dim = 64
        num_patches = (self.img_size[1] // patch_size) * (self.img_size[0] // patch_size)
        
        # Define input layer
        inputs = layers.Input(shape=(self.img_size[1], self.img_size[0], 3))
        
        # Create patches
        # Reshape the input to [batch_size, num_patches, patch_size * patch_size * 3]
        patches = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )(inputs)
        patches = layers.Reshape((num_patches, projection_dim))(patches)
        
        # Add positional embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
        patches = layers.Add()([patches, position_embedding])
        
        # Create transformer blocks
        for _ in range(4):
            # Layer normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(patches)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=4, key_dim=projection_dim // 4, dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add()([attention_output, patches])
            
            # Layer normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = layers.Dense(units=projection_dim * 2, activation=tf.nn.gelu)(x3)
            x3 = layers.Dropout(0.1)(x3)
            x3 = layers.Dense(units=projection_dim)(x3)
            x3 = layers.Dropout(0.1)(x3)
            
            # Skip connection 2
            patches = layers.Add()([x3, x2])
        
        # Classification head
        representation = layers.LayerNormalization(epsilon=1e-6)(patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        representation = layers.Dropout(0.3)(representation)
        
        # Add MLP
        features = layers.Dense(512, activation="relu")(representation)
        features = layers.Dropout(0.3)(features)
        
        # Final classification layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(features)
        
        # Define and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
        
    def train(self, train_dataset, val_dataset, epochs=50, early_stopping_patience=10):
        """Train the model with early stopping"""
        # Create callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                filepath=self.model_dir / 'model_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks_list
        )
        
        # Save the final model
        self.model.save(self.model_dir / 'final_model.h5')
        
        # Save training history
        with open(self.model_dir / 'training_history.json', 'w') as f:
            history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        return history
    
    def evaluate(self, test_dataset):
        """Evaluate the model on test data"""
        results = self.model.evaluate(test_dataset)
        
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        
        return results
    
    def plot_training_history(self, history):
        """Plot training history"""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png')
        plt.close()
    
    def predict_hieroglyph(self, image_path, processor):
        """Predict hieroglyph class for a given image"""
        # Load and preprocess the image
        img = processor.preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Get prediction
        pred = self.model.predict(img)[0]
        pred_class_idx = np.argmax(pred)
        pred_confidence = pred[pred_class_idx]
        
        # Map to Gardiner code
        pred_gardiner_code = processor.inv_class_mapping[pred_class_idx]
        
        return {
            'gardiner_code': pred_gardiner_code,
            'confidence': float(pred_confidence),
            'class_idx': int(pred_class_idx)
        }
    
    def visualize_predictions(self, test_df, processor, num_samples=10):
        """Visualize model predictions on random test samples"""
        # Sample random images
        samples = test_df.sample(num_samples)
        
        rows = (num_samples + 4) // 5
        cols = min(5, num_samples)
        
        plt.figure(figsize=(cols * 3, rows * 3))
        
        for i, (_, row) in enumerate(samples.iterrows()):
            # Get image and true label
            img_path = row['path']
            true_label = row['gardiner_code']
            
            # Get prediction
            pred_result = self.predict_hieroglyph(img_path, processor)
            pred_label = pred_result['gardiner_code']
            confidence = pred_result['confidence']
            
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
        plt.savefig(self.model_dir / 'prediction_results.png')
        plt.close()
    
    def export_for_deployment(self, export_dir='deployment_model'):
        """Export the model for deployment"""
        export_path = Path(export_dir)
        os.makedirs(export_path, exist_ok=True)
        
        # Save in TensorFlow SavedModel format (default)
        tf.saved_model.save(self.model, str(export_path / 'saved_model'))
        
        # Save as TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        with open(export_path / 'model.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print(f"Model exported to {export_dir} in both SavedModel and TFLite formats") 
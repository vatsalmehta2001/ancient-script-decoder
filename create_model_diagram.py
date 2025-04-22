#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a visualization of the advanced hierarchical model architecture
with CNN, EfficientNet, and Vision Transformer components.
"""

import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models, applications
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as path_effects

def create_model_architecture_diagram():
    """Create a visual diagram of the model architecture"""
    # Create directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Set up the figure with a light gray background
    plt.figure(figsize=(18, 14))
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    # Add a title
    title = plt.title('Advanced Hieroglyph Recognition Model Architecture (82.66% Accuracy)', 
                     fontsize=22, fontweight='bold', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Define components and their positions
    components = {
        'input': (0.5, 0.9, 'Input Image\n224×224×3', '#d1e8ff'),
        
        # CNN Branch
        'cnn_start': (0.25, 0.8, 'CNN Branch', '#ffcccc'),
        'cnn_conv1': (0.25, 0.75, 'Conv2D Blocks\n+ Batch Norm', '#ffcccc'),
        'cnn_resid': (0.25, 0.7, 'Residual Blocks', '#ffcccc'),
        'cnn_se': (0.25, 0.65, 'Squeeze-Excite\nBlocks', '#ffcccc'),
        'cnn_pool': (0.25, 0.6, 'Global Pooling\n(Avg + Max)', '#ffcccc'),
        'cnn_out': (0.25, 0.55, 'CNN Features\n1024', '#ffcccc'),
        
        # EfficientNet Branch
        'eff_start': (0.5, 0.8, 'EfficientNetV2S', '#ccffcc'),
        'eff_freeze': (0.5, 0.75, 'Frozen Layers\n(70%)', '#ccffcc'),
        'eff_train': (0.5, 0.7, 'Trainable Layers\n(30%)', '#ccffcc'),
        'eff_pool': (0.5, 0.65, 'Global Pooling\n(Avg + Max)', '#ccffcc'),
        'eff_out': (0.5, 0.6, 'EfficientNet\nFeatures\n1280', '#ccffcc'),
        
        # Vision Transformer Branch
        'vit_start': (0.75, 0.8, 'Vision Transformer', '#ccccff'),
        'vit_patch': (0.75, 0.75, 'Patch Embedding\n(8×8 patches)', '#ccccff'),
        'vit_pos': (0.75, 0.7, 'Position Embedding', '#ccccff'),
        'vit_class': (0.75, 0.65, 'Class Token + Attention\n(8 Transformer Blocks)', '#ccccff'),
        'vit_out': (0.75, 0.6, 'ViT Features\n768', '#ccccff'),
        
        # Fusion and Classification
        'fusion': (0.5, 0.5, 'Feature Fusion Layer\nConcatenation + Dense (1024)', '#ffffcc'),
        'dropout1': (0.5, 0.45, 'Dropout (0.3)', '#f0f0f0'),
        'dense1': (0.5, 0.4, 'Dense Layer (512)\n+ Batch Norm', '#ffffcc'),
        'dropout2': (0.5, 0.35, 'Dropout (0.2)', '#f0f0f0'),
        'output': (0.5, 0.3, 'Output Layer\n(310 classes)', '#ffd1c1'),
        'softmax': (0.5, 0.25, 'Softmax Activation', '#ffd1c1'),
        'hieroglyph': (0.5, 0.15, 'Hieroglyph Classification\n(Gardiner Codes)', '#ffe0b2')
    }
    
    # Draw boxes for each component
    box_width, box_height = 0.15, 0.04
    for name, (x, y, label, color) in components.items():
        rect = Rectangle((x - box_width/2, y - box_height/2), box_width, box_height, 
                        facecolor=color, edgecolor='black', alpha=0.9, zorder=1)
        ax.add_patch(rect)
        
        # Add multi-line text
        lines = label.split('\n')
        line_height = box_height / (len(lines) + 1)
        
        for i, line in enumerate(lines):
            y_text = y - (len(lines) - 1) * line_height / 2 + i * line_height
            text = plt.text(x, y_text, line, ha='center', va='center', 
                           fontsize=10, fontweight='bold', zorder=2)
            text.set_path_effects([path_effects.withStroke(linewidth=1, foreground='white')])
    
    # Draw connections
    arrows = [
        # Input to branches
        ('input', 'cnn_start'),
        ('input', 'eff_start'),
        ('input', 'vit_start'),
        
        # CNN branch
        ('cnn_start', 'cnn_conv1'),
        ('cnn_conv1', 'cnn_resid'),
        ('cnn_resid', 'cnn_se'),
        ('cnn_se', 'cnn_pool'),
        ('cnn_pool', 'cnn_out'),
        
        # EfficientNet branch
        ('eff_start', 'eff_freeze'),
        ('eff_freeze', 'eff_train'),
        ('eff_train', 'eff_pool'),
        ('eff_pool', 'eff_out'),
        
        # Vision Transformer branch
        ('vit_start', 'vit_patch'),
        ('vit_patch', 'vit_pos'),
        ('vit_pos', 'vit_class'),
        ('vit_class', 'vit_out'),
        
        # Feature fusion and classification
        ('cnn_out', 'fusion'),
        ('eff_out', 'fusion'),
        ('vit_out', 'fusion'),
        ('fusion', 'dropout1'),
        ('dropout1', 'dense1'),
        ('dense1', 'dropout2'),
        ('dropout2', 'output'),
        ('output', 'softmax'),
        ('softmax', 'hieroglyph')
    ]
    
    for start, end in arrows:
        start_x, start_y = components[start][0], components[start][1] - box_height/2
        end_x, end_y = components[end][0], components[end][1] + box_height/2
        
        # If connecting to fusion from branches, adjust paths
        if end == 'fusion' and start in ['cnn_out', 'vit_out']:
            arrow = FancyArrowPatch(
                (start_x, start_y), 
                (end_x, end_y),
                connectionstyle=f"arc3,rad=0.2", 
                arrowstyle='-|>', 
                mutation_scale=15,
                linewidth=1.5, 
                color='#555555', 
                zorder=0
            )
        else:
            arrow = FancyArrowPatch(
                (start_x, start_y), 
                (end_x, end_y),
                arrowstyle='-|>', 
                mutation_scale=15,
                linewidth=1.5, 
                color='#555555', 
                zorder=0
            )
        ax.add_patch(arrow)
    
    # Add legend for the branches
    legend_items = [
        Rectangle((0, 0), 1, 1, facecolor='#ffcccc', edgecolor='black', alpha=0.9),
        Rectangle((0, 0), 1, 1, facecolor='#ccffcc', edgecolor='black', alpha=0.9),
        Rectangle((0, 0), 1, 1, facecolor='#ccccff', edgecolor='black', alpha=0.9),
        Rectangle((0, 0), 1, 1, facecolor='#ffffcc', edgecolor='black', alpha=0.9),
        Rectangle((0, 0), 1, 1, facecolor='#ffd1c1', edgecolor='black', alpha=0.9)
    ]
    
    legend_labels = [
        'CNN Branch with Residual Blocks & SE',
        'EfficientNetV2S Transfer Learning',
        'Vision Transformer with Self-Attention',
        'Feature Fusion & Hidden Layers',
        'Classification Output'
    ]
    
    plt.legend(legend_items, legend_labels, loc='lower center', 
              bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=12)
    
    # Remove axis ticks and labels
    plt.axis('off')
    
    # Add technical specifications in a box at the top-right with more space
    specs_text = (
        "Technical Details:\n"
        "- Input: 224×224×3 RGB images\n"
        "- CNN: Custom residual network with squeeze-excite blocks\n"
        "- EfficientNetV2S: Pre-trained on ImageNet with 70% frozen layers\n"
        "- ViT: 8×8 patch size, 8 transformer layers, 6 attention heads\n"
        "- Training: Cosine decay with warmup, label smoothing, mixed precision\n"
        "- Performance: 82.66% top-1 accuracy, 95% top-3 accuracy"
    )
    
    # Create a more distinct box positioned at the top right with better spacing
    plt.figtext(0.77, 0.92, specs_text, ha='left', va='top', 
             fontsize=11, fontweight='medium', 
             bbox=dict(facecolor='#f8f8f8', alpha=0.95, edgecolor='#aaaaaa', 
                      boxstyle='round,pad=0.7', linewidth=1.5))
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('images/model_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model architecture diagram created successfully: images/model_architecture_diagram.png")

if __name__ == "__main__":
    create_model_architecture_diagram() 
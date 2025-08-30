# SketchRNN Transformer - Neural Sketch Generation

## Overview
This project implements a novel Transformer-based approach to sketch generation, inspired by Google's SketchRNN but leveraging the power of attention mechanisms for sequential drawing prediction. The model learns to generate sketches from class labels using the QuickDraw dataset, producing vector-based drawings through coordinate prediction and pen state control.

## Key Innovation
Unlike the original SketchRNN which uses LSTM-based sequence modeling, this implementation employs a **Transformer decoder architecture** with specialized branches for different drawing aspects:
- **Multi-head attention** for capturing long-range dependencies in stroke sequences
- **Dual-output branches** for offset prediction and pen state classification
- **Class-conditioned generation** using embedded class tokens
- **Positional encoding** for understanding drawing progression

## Architecture Components

### 1. Transformer Decoder (`Decoder` class)
**Multi-Branch Architecture:**
- **Base Blocks (Nb)**: Shared feature extraction using self-attention
- **Offset Branch (No)**: Predicts x,y coordinate offsets for pen movement
- **Pen State Branch (Np)**: Classifies pen states (down, up, end) for drawing control

**Key Features:**
- **Class Embedding**: Dense layer projection of one-hot class labels
- **Positional Encoding**: Sinusoidal encoding for sequence position awareness
- **Multi-Head Attention**: Parallel attention heads for diverse feature capture
- **Layer Normalization**: Stabilizes training and improves convergence

### 2. Multi-Head Attention (`MultiHeadAttention` class)
**Scaled Dot-Product Attention:**
- Query, Key, Value projections with learnable weights
- Attention weight computation with temperature scaling
- Look-ahead masking for autoregressive generation
- Multi-head parallel processing for rich representation learning

### 3. Decoder Block (`DecoderBlock` class)
**Transformer Layer Components:**
- Self-attention with residual connections
- Feed-forward networks with ReLU activation
- Layer normalization and dropout for regularization
- Skip connections for gradient flow optimization

## Data Representation

### Stroke Format (5-dimensional vectors)
```
[dx, dy, pen_down, pen_up, end_stroke]
```
- **dx, dy**: Normalized coordinate offsets (-1 to 1)
- **pen_down**: Binary flag for drawing state
- **pen_up**: Binary flag for pen lift
- **end_stroke**: Binary flag for drawing termination

### Data Processing Pipeline
1. **Coordinate Normalization**: Min-max scaling to [0,1] range
2. **Stroke Vectorization**: Conversion to 5D representation
3. **Sequence Padding**: Fixed-length sequences with padding tokens
4. **Class Encoding**: One-hot encoding for sketch categories

## Training Methodology

### Loss Functions
- **Offset Loss**: Mean Squared Error for coordinate regression
- **Pen State Loss**: Categorical Cross-Entropy for state classification
- **Combined Optimization**: Joint training with unified gradient updates

### Training Configuration
```python
Architecture: 8 base + 4 offset + 4 pen blocks
Model Dimension: 256
Attention Heads: 8
Hidden Units: 1024
Sequence Length: 100
Batch Size: 64
Learning Rate: 0.0001
```

### Optimization Strategy
- **Adam Optimizer**: Adaptive learning rate with momentum
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Model checkpointing based on loss improvement
- **Look-Ahead Masking**: Ensures autoregressive property during training

## Generation Process

### Autoregressive Sketch Creation
1. **Initialization**: Start with class embedding token
2. **Sequential Prediction**: Generate strokes one at a time
3. **Condition on History**: Use previous strokes for context
4. **Termination**: Stop when end token is predicted or max length reached

### Generation Parameters
- **Temperature Control**: Sampling randomness adjustment
- **Max Steps**: Maximum drawing sequence length
- **Class Conditioning**: Sketch category specification
- **Beam Search**: Optional multiple hypothesis generation

## Dataset Integration

### QuickDraw Dataset Processing
- **Multi-Class Support**: Handles multiple sketch categories simultaneously
- **Sample Balancing**: Configurable samples per class (30K default)
- **Data Augmentation**: Coordinate normalization and sequence padding
- **Batch Processing**: Efficient mini-batch training pipeline

### Supported Categories
Successfully tested on sketch classes including:
- Geometric shapes (zigzag, lightning)
- Animals (cat, dog, bird)
- Objects (car, house, tree)
- Abstract concepts (smile, star)

## Results and Performance

### Generation Quality
- **Comparable to SketchRNN**: Achieves similar sketch quality to original implementation
- **Class Consistency**: Generated sketches match target categories accurately
- **Stroke Smoothness**: Natural pen movement patterns with proper connectivity
- **Diversity**: Multiple variations possible for same class

### Computational Efficiency
- **GPU Acceleration**: Optimized for CUDA-enabled training
- **Parallel Attention**: Faster than sequential LSTM processing
- **Memory Management**: Efficient handling of sequence padding
- **Batch Generation**: Simultaneous multi-sketch production

## Technical Implementation

### Key Functions
- **`positional_encoding()`**: Sinusoidal position embeddings
- **`sdp_attention()`**: Scaled dot-product attention mechanism
- **`to_big_strokes()`**: Stroke normalization and vectorization
- **`generate_sketch()`**: Autoregressive sketch generation
- **`render_sketch()`**: Visualization of generated drawings

### Model Architecture
```
Input: Class Label + Previous Strokes
↓
Class Embedding + Position Encoding
↓
Base Transformer Blocks (8x)
↓
┌─ Offset Branch (4x) → Coordinate Prediction
└─ Pen State Branch (4x) → State Classification
↓
Output: Next Stroke (dx, dy, pen_state)
```

## Usage Instructions

### Training
```python
model, offset_losses, pen_losses = train_model(
    Nb=8, No=4, Np=4,           # Architecture blocks
    dm=256, h=8, hidden=1024,   # Model dimensions
    max_len=100, batch_size=64, # Sequence parameters
    epochs=50, filepath=data_paths
)
```

### Generation
```python
sketch = generate_sketch(
    model=trained_model,
    label=class_index,
    max_len=100,
    max_steps=250
)
render_sketch(sketch)
```

## Advantages Over Original SketchRNN

### Architectural Benefits
- **Parallel Processing**: Attention mechanisms enable parallel computation
- **Long-Range Dependencies**: Better capture of global sketch structure
- **Flexible Conditioning**: Easy integration of additional context
- **Scalable Architecture**: Modular design for easy modification

### Performance Improvements
- **Training Efficiency**: Faster convergence with attention mechanisms
- **Generation Quality**: More coherent long sequences
- **Class Separation**: Better category-specific feature learning
- **Extensibility**: Easy addition of new sketch classes

## Technical Requirements
- **TensorFlow 2.x**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Sketch visualization
- **CUDA GPU**: Recommended for training acceleration
- **Memory**: 8GB+ RAM for large datasets

## Applications
- **Digital Art Creation**: Automated sketch generation for artists
- **Educational Tools**: Teaching drawing fundamentals
- **Game Development**: Procedural sketch generation for games
- **Design Assistance**: Quick concept sketching for designers
- **Research**: Understanding human drawing patterns

This Transformer-based approach represents a significant advancement in neural sketch generation, combining the power of attention mechanisms with the intuitive stroke-based representation of drawings, resulting in a more efficient and effective sketch generation system.

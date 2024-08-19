# Predicting Psychosis vs Healthy Brain with Computer Vision

## Project Overview
This project uses advanced computer vision and deep learning techniques to predict psychosis from brain imaging data. We compare three different neural network architectures to classify brain scans as either psychotic or healthy.

## Models Implemented

### 1. FEP_HC Model
- **Architecture**: 5 convolutional layers, 3 fully connected layers, LeakyReLU activation, Dropout (p=0.3), BatchNorm3d, BatchNorm1d
- **Training**: AdamW optimizer (lr=1e-5), ReduceLROnPlateau scheduler, 20 epochs, Class weights
- **Performance**: 
  - Best validation accuracy: 75.57%
  - Final validation AUC: 0.7698
- **Confusion Matrices**:
  - Training: [[355,0],[0,178]]
  - Validation: [[65,11],[21,34]]

### 2. AlexNet 3D
- **Architecture**: 5 convolutional layers, 3 fully connected layers, LeakyReLU activation, Dropout (p=0.5 and 0.3), BatchNorm3d, BatchNorm1d
- **Training**: AdamW optimizer (lr=1e-5), ReduceLROnPlateau scheduler, 20 epochs, Class weights
- **Performance**:
  - Best validation accuracy: 73.28%
  - Final validation AUC: 0.7481
- **Confusion Matrices**:
  - Training: [[352,3],[14,164]]
  - Validation: [[61,25],[15,30]]

### 3. VoxCNN
- **Architecture**: 4 convolutional blocks (multiple conv layers), 3 fully connected layers, LeakyReLU activation, Dropout (p=0.5), BatchNorm3d, BatchNorm1d
- **Training**: AdamW optimizer (lr=1e-5), ReduceLROnPlateau scheduler, 20 epochs, Class weights
- **Performance**:
  - Best validation accuracy: 70.23%
  - Final validation AUC: 0.7481
- **Confusion Matrices**:
  - Training: [[352,3],[104,74]]
  - Validation: [[60,13],[26,32]]

## Key Findings
- FEP_HC model shows the highest validation accuracy and AUC.
- AlexNet 3D and VoxCNN models demonstrate better generalization to unseen data.
- All models show some level of overfitting, with FEP_HC exhibiting the most pronounced effect.
- Class imbalance handling has improved across all models, likely due to class weights in the loss function.

## Conclusion
The choice of model depends on specific requirements:
- For peak accuracy: FEP_HC model
- For better generalization: AlexNet 3D or VoxCNN
- Consider trade-offs between performance, inference speed, and model size for deployment.

## Future Work
- Explore ensemble methods to combine model strengths
- Investigate techniques to further reduce overfitting
- Collect more diverse data to improve generalization
- Optimize models for deployment in clinical settings

## Technologies Used
- Python
- PyTorch
- Scikit-learn
- Matplotlib (for visualization)


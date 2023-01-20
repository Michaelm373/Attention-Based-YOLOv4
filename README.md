# Attention-Based-YOLOv4
This is my implementation of AS-YOLO, which is an improved YOLOv4 based on Attention Mechanism 

Link: https://ieeexplore.ieee.org/document/9390855

The model was simplified by removing the squeezenet layers and the CSPDarknet backbone was switched out for the convolutional layers of a pre-trained VGG-19 network

The attention mechanism (CBAM https://arxiv.org/abs/1807.06521) which using channel and spatial attention modules was also switched out for ECA-CBAM (https://dl.acm.org/doi/fullHtml/10.1145/3529466.3529468) which uses an efficient channel attention module and outperforms standard CBAM attention modules in speed and accuracy. 

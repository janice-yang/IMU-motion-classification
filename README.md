# Classification of human motion using a CNN-LSTM learning model
Submission for the 2025 MathWorks AI Challenge - Human Motion Recognition Using IMUs, July 2025, by Jie Yang. This work is a personal interest and training project, and was not funded or sponsored.

## CNN-LSTM model trained on the UCI HAR dataset

(PLACEHOLDER for CNN-LSTM background & rationale) 
- CNN for learning positional patterns from sensor data
- LSTM for learning motion sequence patterns
Prior works featuring LSTM or combined architectures for IMU motion classification:
- Marcos Mazon, Daniel, Marc Groefsema, Lambert R. B. Schomaker, and Raffaella Carloni. “IMU-Based Classification of Locomotion Modes, Transitions, and Gait Phases with Convolutional Recurrent Neural Networks.” Sensors 22, no. 22 (January 2022): 8871. https://doi.org/10.3390/s22228871.
- Pesenti, Mattia, Giovanni Invernizzi, Julie Mazzella, Marco Bocciolone, Alessandra Pedrocchi, and Marta Gandolla. “IMU-Based Human Activity Recognition and Payload Classification for Low-Back Exoskeletons.” Scientific Reports 13, no. 1 (January 21, 2023): 1184. https://doi.org/10.1038/s41598-023-28195-x.
- Sherratt, Freddie, Andrew Plummer, and Pejman Iravani. “Understanding LSTM Network Behaviour of IMU-Based Locomotion Mode Recognition for Applications in Prostheses and Wearables.” Sensors 21, no. 4 (January 2021): 1264. https://doi.org/10.3390/s21041264.
- Zhang, Zhibo, Yanjun Zhu, Rahul Rai, and David Doermann. “PIMNet: Physics-Infused Neural Network for Human Motion Prediction.” IEEE Robotics and Automation Letters 7, no. 4 (October 2022): 8949–55. https://doi.org/10.1109/LRA.2022.3188892.


`main_livescript.mlx` contains instructions and code for:
1. Extracting and processing training data from UCI HAR dataset:
- Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K.
- Training and test data available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Building and training a CNN-LSTM model with 5-fold cross-validation
- (PLACEHOLDER for model architecture & training strategy diagrams)
3. Evaluating and visualizing the trained model's performance on held-out test dataset
*Additional datasets for model evaluation*
4. Extract and prepare HARTH dataset:
- Logacjov, A., Bach, K., Kongsvold, A., Bårdstu, H. B., & Mork, P. J. (2021). HARTH: A Human Activity Recognition Dataset for Machine Learning. Sensors, 21(23), 7853. https://doi.org/10.3390/s21237853
- Data available at https://github.com/ntnu-ai-lab/harth-ml-experiments
5. Simulating a new dataset of IMU sensor readings using imuSensor (Navigation Toolbox, Sensor Fusion and Tracking Toolbox).
6. Evaluating and visualizing the trained model's performance on the new datasets from **4** and **5**.

`datasets` folder stores training and test data extracted from the above sources (or other user-defined sources)

`models` folder stores trained models and their evaluation results

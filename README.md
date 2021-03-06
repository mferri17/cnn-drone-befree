# CNN Drone BeFree

Code repository for my Master Thesis:  
*Interpretation of Neural Networks and Advanced Image Augmentation for Visual Control of Drones in Human Proximity*

I have considered [this](https://github.com/idsia-robotics/proximity-quadrotor-learning) work from Dario Mantegazza, later expanded [here](https://github.com/FullMetalNicky/FrontNetPorting) by Nicky Zimmerman. The aims of the project is to make the drone flying outside of the room arena in which the dataset has been collected. For doing so, I am using network interpretability techniques together with background replacement/randomization on the dataset images.

- Check out the Jupyter Notebooks in `notebooks` for a better understanding of the initial work on interpretability (GradCAM) and human segmentation (MaskRCNN)
- Check out the Python Scripts in `src` for a better understanding of the training procedure conducted through TensorFlow and [tf.data](https://www.tensorflow.org/guide/data) on the masked dataset; also testing and visualization scripts are available for use, easily launched by executables in `src/runners`
- Some fast visual results are available in the `results` folder
- Several models trained during the development of the project are available into the `model` folder
- The best model so far with both quantitative and qualitative results is available [here](https://github.com/mferri17/cnn-drone-befree/tree/main/models/20210112%20final%20comparison)
- My thesis report is available on https://github.com/mferri17/thesis-master


## Deliverables

- [Thesis summary](https://github.com/mferri17/thesis-master/blob/main/deliverable/Thesis%20Summary.pdf) (5 pages)
- [Final thesis](https://github.com/mferri17/thesis-master/blob/main/deliverable/Ferri_Marco_Tesi_LM_22febbraio2021.pdf)
- [Final presentation](https://github.com/mferri17/thesis-master/blob/main/presentation/Ferri_Marco_Presentazione_LM_22febbraio2021.pdf)
- [Demonstrational videos](https://www.youtube.com/playlist?list=PLbpVnR3zJG9rVIbU_VIxRt1VhcCe53Cqo)


## Abstract

We consider the task of predicting the pose of a person moving in front of a drone, using the input images coming from an on-board camera. We aim to improve a machine learning model designed for this intent [[Mantegazza et al., 2019](https://github.com/idsia-robotics/proximity-quadrotor-learning)]. The approach relies on supervised learning to perform a regression on the user’s pose through a Residual Neural Network. The training data is collected in a dedicated drone arena, using a Motion Capture system to acquire the ground truth. The prototype achieves good performance inside the arena but cannot fulfill its duty in unknown environments.

First, we understand the main issues of the learned task through network interpretation. Applying Grad-CAM [[Selvaraju et al., 2019](https://arxiv.org/abs/1610.02391)], we observe that the model not only focus on the user who is actually facing the drone’s camera. Instead, various portions of the input images are considered when the model makes its predictions. We assume that the neural network has undesirably learned some details about the drone arena in which the dataset has been collected.

As a solution, we develop an advanced data augmentation technique designed for enhancing the generalization capabilities of the model. The goal is to break the relationship between the model’s learning and the training room. Our approach consists on modifying the original dataset through images’ background replacement, allowing us to simulate data collected in many different environments. The implementation is done by using Mask R-CNN [[He et al., 2018](https://arxiv.org/abs/1703.06870)] to compute the user’s mask from each sample in the original training set. Then, we use the computed masks for replacing the background of the corresponding images and retraining the neural network.
We run experiments that show that our proposal is successful both from quantitative and qualitative viewpoints. The new model, trained on the augmented dataset, produces satisfactory results in a large variety of real-world scenarios.





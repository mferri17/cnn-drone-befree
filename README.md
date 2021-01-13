# CNN Drone BeFree

Code repository for my Master Thesis project about visual end-to-end control of drones.

I am working out on [this](https://github.com/idsia-robotics/proximity-quadrotor-learning) work from Dario Mantegazza, later expanded [here](https://github.com/FullMetalNicky/FrontNetPorting) by Nicky Zimmerman. The aims of the project is to make the drone flying outside of the room arena in which the dataset has been collected. For doing so, I am using network interpretability techniques together with background replacement/randomization on the dataset images.

- Check out the Jupyter Notebooks in `notebooks` for a better understanding of the initial work on interpretability (GradCAM) and human segmentation (MaskRCNN)
- Check out the Python Scripts in `src` for a better understanding of the training procedure conducted through TensorFlow and [tf.data](https://www.tensorflow.org/guide/data) on the masked dataset; also testing and visualization scripts are available for use, easily launched by executables in `src/runners`
- Some fast visual results are available in the `results` folder
- Several models trained during the development of the project are available into the `model` folder
- The best model so far with both quantitative and qualitative results is available [here](https://github.com/mferri17/cnn-drone-befree/tree/main/models/20210112%20final%20comparison)
- My thesis report is available on https://github.com/mferri17/thesis-master



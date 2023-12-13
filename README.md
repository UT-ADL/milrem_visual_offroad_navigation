# Vision-based off-road navigation with geographical hints

## Summary

| Company name | [Milrem Robotics]([https://website.link](https://milremrobotics.com/)) |
| :--- | :--- |
| Development Team Lead Name | [Tambet Matiisen](https://www.etis.ee/CV/Tambet_Matiisen/eng), University of Tartu |

## Objectives of the Demonstration Project

The goal of the project is to collect and validate dataset for vision-based off-road navigation with geographical hints.

Milrem UGV needs to be able to navigate
1. in unstructured environment (no buildings, roads or other landmarks),
2. with passive sensors (using only camera and GNSS, active sensors make the UGV discoverable to the enemy),
3. with no prior map or with outdated map,
4. with unreliable satellite positioning signals.

System that satisfies the above goals was proposed in the [ViKiNG paper](https://sites.google.com/view/viking-release) by Dhruv Shah and Sergey Levine from University of California, Berkeley. The paper demonstrated vision-based kilometer-scale navigation with geographical hints in semi-structured urban environments, including parks. The goal of this project was to extend the ViKiNG solution to unstructured off-road environments, for example forests.

## Activities and results of demonstration project

### Challenge adressed

The goal of using passive sensors means that the camera is the primary sensor. The currently best known way to make sense of camera images is to use artificial neural networks. These networks need a lot of training data to work well. Therefore the main goal of this project was to collect and validate the data to train artificial neural networks for vision-based navigation.

We set ourselves a goal to collect 50 hours of data consisting of 150 km of trajectories. This was inspired by the ViKiNG paper having 42 hours of training data. Time-wise this goal was achieved, distance-wise slightly less data was collected, 104 km.

In addition to collecting the data we wanted to also validate if it is usable for training the neural networks. We actually went further than that by not only training the networks, but also implementing a proof-of-concept navigation system on [Jackal robot](https://clearpathrobotics.com/jackal-small-unmanned-ground-vehicle/).

### Data sources

The data was collected during Apr 12 - Oct 6, 2023 from 27 orienteering events and 20 self-guided sessions. Details of the places and weather conditions can be found in [this table](https://docs.google.com/spreadsheets/d/1QvA2ZYTeZOpk7b1DCHypi17wS-ywxv5n0ifdOzRoi_o/edit?usp=sharing).

Data collection was performed with golf trolley fitted with following sensors:
* [ZED 2i](https://www.stereolabs.com/products/zed-2) stereo camera
* [Xsens MTI-710G](https://www.movella.com/products/sensor-modules/xsens-mti-g-710-gnss-ins) GNSS/INS device
* 3x [GoPro cameras](https://gopro.com/en/us/shop/cameras/hero12-black/CHDHX-121-master.html) at three different heights

Four different types of data was collected:
1. camera images,
2. visual odometry (trajectories derived from camera movement),
3. GPS trajectories,
4. maps.

Following types of maps were acquired:
* orienteering maps (usually from organizers, sometimes from [Estonian O-Map](https://okaart.osport.ee/))

  ![orienteering map](images/otepaa_orienteering.jpg)
* [Estonian base map](https://geoportaal.maaamet.ee/eng/Spatial-Data/Topographic-Maps/Estonian-Basic-Map-1-10-000-p306.html) (from [Estonian Land Board](https://maaamet.ee/en))

  ![Estonian base map](images/otepaa_base.jpg)
* [Estonian base map](https://geoportaal.maaamet.ee/eng/Spatial-Data/Topographic-Maps/Estonian-Basic-Map-1-10-000-p306.html) with elevation (from [Estonian Land Board](https://maaamet.ee/en))

  ![Estonian base map with elevation](images/otepaa_baseelev.jpg)
* [Estonian orthophoto](https://geoportaal.maaamet.ee/eng/Spatial-Data/Orthophotos-p309.html) (from [Estonian Land Board](https://maaamet.ee/en))

  ![Estonian orthophoto](images/otepaa_orthophoto.jpg)
* Google satellite photo (from [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/start))

  ![Google satellite photo](images/otepaa_satellite.jpg)
* Google road map (from [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/start))

  ![Google road map](images/otepaa_roadmap.jpg)
* Google hybrid map (from [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/start))

  ![Google hybrid map](images/otepaa_hybrid.jpg)

Further cleaning was applied to the data with following sections removed:
* Missing odometry data
* Big change in position: >1.0m per timestep
* Low velocity: <0.05 m/s
* High velocity: >2.5 m/s
* Model prediction errors were analyzed
* Bad trajectories
* Missing or bad camera images
* 90° turns
* Climbing over fallen trees

Altogether this resulted in 94.4 km of trajectories used for training.

In addition the dataset for local planner was combined with [RECON dataset](https://sites.google.com/view/recon-robot/dataset) of 40 hours of autonomously collected trajectories.

### Description of AI technology

The system makes use of two neural networks: local planner and global planner.

**Local planner** takes a camera image and predicts next waypoints, where the robot can drive without hitting obstacles. 
* Inputs to the model are:
  * Current camera image
  * Past 5 camera images
  * Goal image
* Outputs of the model are:
  * Trajectory of 5 waypoints
  * Temporal distance to the goal

The local planner is trained using camera images and visual odometry.

![Local planner](images/local_planner.jpg)

**Global planner** takes the waypoints proposed by the local planner and estimates which of them takes fastest to the final goal.

* Inputs to the model are:
  * Overhead map
  * Current location
  * Goal location
  * Candidate waypoint(s)
* Outputs of the model are:
  * Probability that the waypoint is on the path from current location to goal

The global planner is trained using maps and GPS trajectories.

![Global planner](images/global_planner.jpg)

These two models work in coordination to handle outdated maps and inaccurate GPS:
* as long as the local planner proposes valid waypoints the robot never collides with obstacles,
* as the global planner picks waypoints which are closer to the final destination, it tends to move towards the final goal, even if the GPS positioning is wrong or the map is outdated.

### Results of validation

#### Local planner

For local planner following network architectures were considered:
| Model | Pretrained weights | Trained or finetuned | On-policy tested | Generative | Waypoint proposal method |
|-------|--------------------|----------------------|------------------|------------|--------------------------|
| [VAE](https://sites.google.com/view/viking-release) | - | + | + | + | Sampling from latent representation |
| [GNM](https://sites.google.com/view/drive-any-robot) | + | + | + | - | Cropping the current observation |
| [ViNT](https://general-navigation-models.github.io/vint/index.html)  | + | - | + | + | Goal image diffusion |
| [NoMaD](https://general-navigation-models.github.io/nomad/index.html) | + | - | - | + | Trajectory diffusion |

VAE model was trained from scratch, all other models were used with pre-trained weights from Berkeley group. GNM model was additionally fine-tuned with our own dataset.

The models were tested both off-policy and on-policy. Off-policy means that the model was applied to recorded data, the model's actions were just visualized, but not taken into account. On-policy means that the model’s actions were actually taken on the robot.

**Off-policy results**

* GNM finetuned

  [![GNM finetuned](https://img.youtube.com/vi/PeYGA85I2FI/hqdefault.jpg)](https://youtu.be/PeYGA85I2FI)

* ViNT

  [![ViNT](https://img.youtube.com/vi/pnftnew_JVo/hqdefault.jpg)](https://youtu.be/pnftnew_JVo)

* NoMaD with moving goal

  [![NoMaD with moving goal](https://img.youtube.com/vi/KI7kkKAnis8/hqdefault.jpg)](https://youtu.be/KI7kkKAnis8)

* NoMaD with fixed goal

  [![NoMaD with fixed goal](https://img.youtube.com/vi/xCyGxyZ0rtA/hqdefault.jpg)](https://youtu.be/xCyGxyZ0rtA)

**On-policy results**

We created a fixed course in Delta park with goal images every 2, 5 or 10 meters and measured the goal success rate at each interval. Basically it shows how well the model can move towards the goal image and detect if it has reached the goal image. The operator intervened when the robot was going completely off the path and guided it back to the track.

Success rate with 2m intervals:
| Model | Number of goal images | Number of interventions | Success rate |
|-------|-----------------------|-------------------------|--------------|
| GNM | 38 | 1 | 86.84 |
| GNM_finetuned | 38 | 0 | 81.58 |

Success rate with 5m intervals:
| Model | Number of goal images | Number of interventions | Success rate |
|-------|-----------------------|-------------------------|--------------|
| GNM_finetuned | 17 | 7 | 100 |
| ViNT | 17 | 7 | 100 |

Success rate with 10m intervals:
| Model | Number of goal images | Number of interventions | Success rate |
|-------|-----------------------|-------------------------|--------------|
| ViNT | 8 | 9 | 100 |

#### Global planner

For global planner following network architectures were considered:
* [contrastive MLP](https://sites.google.com/view/viking-release)
* [U-Net](https://arxiv.org/abs/1505.04597)

As the U-Net approach worked much better, the contrastive approach was abandoned. Most of the experimentation was done with the base map with elevation. 

Following videos show a simulation where the robot proposes a number of random waypoints and then moves towards the one that has the highest probability, i.e. it is on-policy, but simulated.

[![GNM finetuned](https://img.youtube.com/vi/wI3Tavbgs5M/hqdefault.jpg)](https://youtu.be/wI3Tavbgs5M)

### Technical architecture

/copy-paste network architectures from the papers?/

For local planner following network architectures were tried:
* VAE (as in the original [ViKiNG paper](https://sites.google.com/view/viking-release))
* [GNM](https://sites.google.com/view/drive-any-robot)
* [ViNT](https://general-navigation-models.github.io/vint/index.html)
* [NoMaD](https://general-navigation-models.github.io/nomad/index.html)

For global planner following network architectures were tried:
* MLP (as in the original [ViKiNG paper](https://sites.google.com/view/viking-release))
* [U-Net](https://arxiv.org/abs/1505.04597)

### Potential areas of use

The working solution could be used in any area that needs navigation in unstructured environment with poor GPS signal and outdated maps, for example:
* military,
* agriculture,
* rescue.

The particular dataset collected in this project will be used to create visual navigation benchmark and regular international robot orienteering competition. Such competition will make novel solutions and international talent available to Milrem Robotics.

### Lessons learned

* Training neural networks in 2023 is still hard.
* Dataset curation is non-trivial and less documented than model training.
* Should use (or fine-tune) pre-trained models whenever available.
* Off-policy performance (on recordings) does not match on-policy performance (on robot).

### Description of User Interface 

/add visual image/

* The screen shows current camera image and proposed trajectories.
* Top right shows the goal image.
* Bottom right shows the map and probability map (the recommended path from current position to goal)
* Bottom left show top-down view of the trajectory.
* Top left shows the robot command.

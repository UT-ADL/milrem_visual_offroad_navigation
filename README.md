# Vision-based off-road navigation with geographical hints

## Summary

| Company name | [Milrem Robotics]([https://website.link](https://milremrobotics.com/)) |
| :--- | :--- |
| Development Team Lead Name | [Tambet Matiisen](https://www.etis.ee/CV/Tambet_Matiisen/eng), University of Tartu |
| Objectives of the Demonstration Project |  |

## Objectives of the Demonstration Project

The goal of the project is to collect and validate dataset for vision-based off-road navigation with geographical hints.

Milrem UGV needs to be able to navigate
1. in unstructured environment (no buildings, roads, etc.),
2. with passive sensors (using only camera and GNSS, active sensors make the UGV discoverable to the enemy),
3. with no prior map or with outdated map,
4. with unreliable satellite positioning signal.

System that satisfies the above criteria was proposed in [ViKiNG paper](https://sites.google.com/view/viking-release) by Dhruv Shah and Sergey Levine from University of California, Berkeley. The paper demonstrated vision-based kilometer-scale navigation with geographical hints in semi-structured urban environment, including parks. The goal of this project was to extend the ViKiNG solution to unstructured off-road environment, for example forests.

## Activities and results of demonstration project

### Challenge adressed

Vision-based navigation in unstructured environment can only be achieved with the help of AI, in particular with artificial neural networks. Neural networks need a lot of training data to work well. Therefore the main goal of this project was to collect and validate the data to train artificial neural networks. We set ourselves a goal to collect 50 hours of data consisting of 150 km of trajectories. In the end 50 hours of data was collected with slightly lesser distance of 104 km.

In addition to collecting the data we wanted to also validate if it is usable for training the neural networks. We actually went a bit further than that by not only training the networks, but also implementing proof-of-concept system on [Jackal robot](https://clearpathrobotics.com/jackal-small-unmanned-ground-vehicle/).

### Data sources

For the purpose of this project a dataset was collected. The dataset consists of three parts:
1. camera images (and accompanying visual odometry),
2. GPS trajectories,
3. maps.

The data was collected from [27 orienteering events](https://docs.google.com/spreadsheets/d/1QvA2ZYTeZOpk7b1DCHypi17wS-ywxv5n0ifdOzRoi_o/edit?usp=sharing) within and around Tartu. Data collection was performed with golf trolley fitted with following sensors:
* [ZED 2i](https://www.stereolabs.com/products/zed-2) stereo camera
* [Xsens MTI-710G](https://www.movella.com/products/sensor-modules/xsens-mti-g-710-gnss-ins) GNSS/INS device
* 3x [GoPro cameras](https://gopro.com/en/us/shop/cameras/hero12-black/CHDHX-121-master.html) at three different heights

In addition to the sensor readings, following maps of each area were acquired:
* orienteering maps (usually from organizers, sometimes from [Estonian O-Map](https://okaart.osport.ee/))
* [Estonian base map](https://geoportaal.maaamet.ee/eng/Spatial-Data/Topographic-Maps/Estonian-Basic-Map-1-10-000-p306.html) (from [Estonian Land Board](https://maaamet.ee/en))
* [Estonian base map](https://geoportaal.maaamet.ee/eng/Spatial-Data/Topographic-Maps/Estonian-Basic-Map-1-10-000-p306.html) with elevation (from [Estonian Land Board](https://maaamet.ee/en))
* [Estonian orthophoto](https://geoportaal.maaamet.ee/eng/Spatial-Data/Orthophotos-p309.html) (from [Estonian Land Board](https://maaamet.ee/en))
* Google satellite photo (from [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/start))
* Google road map (from [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/start))

### Description of AI technology

/some diagram of interaction between local and global planner?/

The system makes use of two neural networks: local planner and global planner. 
* **Local planner** takes camera image and predicts next waypoints, where the robot can drive without hitting obstacles. This model is trained using camera images and visual odometry.
* **Global planner** takes the waypoints proposed by local planner and estimates which of them takes fastest to the final goal. This model is trained using maps and GPS trajectories.

These two models work in coordination to overcome the outdated maps and inaccurate GPS problem:
* as the local planner does not propose waypoints that collide with obstacles, the robot never gets stuck,
* as the global planner picks waypoints which are closer to the final destination, it tends to move towards final goal, even if the GPS positioning is wrong or map is outdated.

### Results of validation

/add some tables, videos with results, move current text to Lessons learned/

For training the local planner the dataset seemed insufficient or contained too simple trajectories (moving mostly forward). We experimented with following options:
* combining our collected dataset with [RECON dataset](https://sites.google.com/view/recon-robot/dataset)
* fine-tuning existing general navigation models, i.e. [GNM](https://sites.google.com/view/drive-any-robot), [ViNT](https://general-navigation-models.github.io/vint/index.html) and [NoMaD](https://general-navigation-models.github.io/nomad/index.html)

The results were inconculsive, sometimes the fine-tuned model was performing better, sometimes worse. The original general navigation models were also unreliable, they were not always able to avoid the obstacles. More work is needed to make visual navigation reliable. Alternative model outputs could be considered, e.g. predicting free space instead of trajectories and proposing waypoints from that free space. Also collection of more explorative data directly with robot might be necessary.

Global planner trained much better and was able to estimate reasonably well the recommended path between two points. We also observed different behavior for different map modalities, e.g. base map and orthophotos. More work is needed to reduce the artifacts produced by the fully convolutional network and some map modalities might need further tuning.

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

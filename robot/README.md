# Robot

This repo contains the code to run the local planner and the global planner on the robot.

To run the local and global planner, a working Robot Operating System (ROS) environment is required. A local installation of ROS is recommended which can be done by following this link : ['ROS desktop installation'](http://wiki.ros.org/noetic/Installation/Ubuntu)

## Building

```
mkdir -p milrem_ws/src
cd milrem_ws/src
git clone git@gitlab.cs.ut.ee:milrem/robot.git
catkin build
```
Here, the catkin_ws is `milrem_ws`   

## Environment
Everything in the robot (or with ros bags) use the ONNX models. To run the models and all other packages, a using a dedicated conda environment is recommended.   
To create this environment, please go through the following steps:   
```
conda install mamba -c conda-forge
conda create -n milrem-aire-robot python=3.9.16
mamba install pytorch-lightning tensorboard pandas jupyter matplotlib pytransform3d transformations h5py utm onnx opencv moviepy -c conda-forge
mamba install pyyaml rasterio pyproj
pip install onnxruntime-gpu==1.16.0
``` 

Or, alternatively the conda environment can be created by:   
```
cd milrem_ws/src/robot
conda env create -f environment-robot.yml
```

After creating the environment, activate the environment. Everything should be good to go now.   
```
conda activate milrem-aire-robot
```

## Installing required ros packages
From the base of the catkin_ws, i.e., `milrem_ws` run the following command:
```
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

## Running
From the base of the catkin_ws, i.e. `milrem_ws`, run the following commands:   
```
# download demo bag
scp username@rocket.hpc.ut.ee:/gpfs/space/projects/Milrem/bagfiles/2023-08-30-17-40-45_10.bag src/robot/data/bags/
roslaunch robot inference_bag.launch
```

### ZED-ros-wrapper
To run the inference with ZED cameras live, the `zed-ros-wrapper` package needs to be installed with required dependencies.   
To install this package and all necessary dependencies, please refer to: [`https://github.com/stereolabs/zed-ros-wrapper`](https://github.com/stereolabs/zed-ros-wrapper) and [`https://www.stereolabs.com/docs/ros/`](https://www.stereolabs.com/docs/ros/)   

Please run the following commands from the base of the `catkin_ws`:   
```
cd src
git submodule add https://github.com/stereolabs/zed-ros-wrapper
cd ..
catkin build
source devel/setup.bash
```


### Spaced Goals (only local planner)
Here, only the local planner is used to get the robot running in a previously driven track. First a rosbag is recorded with all the necessary topics while driving the robot manually over a track.      
Please make sure that the current working directory is `milrem_ws`, then run the following command in terminal:   
```
rosbag record $(cat src/robot/data/topics/topics.txt)
```   
Here `topics.txt` is the txt file where all the topics that needs to be recorded are listed.    
This will create a rosbag inside the catkin_ws.   

Once this is done. it is good to go. Then the robot needs to driven over the track. Once the preferred track is done driving, pressing `ctrl + c` in the terminal where the rosbag is being recorded will stop the recording.   

When the recording is finished, then run the following command making sure that the present working directory is `milrem_ws`:   
```
python3 src/robot/scripts/extraction_img_pose.py --bags_basename YOUR_ROSBAG_PATH --output_dir src/robot/data/bags/extracted_dataset/
```   

This will take some time extracting the data (iamges and odometry position), depending on how big the ros bag is. Once the data is extracted, it will create a directory inside `CATKIN_WS/src/robot/data/extracted_data/` which will have 2 sub-folders: `csv` and `images`, each of which will have relavant data.    


Following the compeltion of this step, run the following command:
```
roslaunch robot inference_spaced_goals.py
```

Or, to record the video:   
```
roslaunch robot inference_spaced_goals.py record_video:=YOUR_VIDEO_OUTPUT_FILE_PATH.mp4
```


### Local + Global Planner
#### Image Crops and a Goal Image

##### With ros bags
```
roslaunch robot inference_bag.launch
```
Or, to record the video:   
```
roslaunch robot inference_bag.launch record_video:=YOUR_VIDEO_OUTPUT_FILE_PATH.mp4
```

##### With live camera
```
roslaunch robot inference.launch
```
Or, to record the video:   
```
roslaunch robot inference.launch record_video:=YOUR_VIDEO_OUTPUT_FILE_PATH.mp4
```

#### Running the NoMaD model
##### With ros bags


```
roslaunch robot inference_nomad_bag.launch
```
Or, to record the video:   
```
roslaunch robot inference_nomad_bag.launch record_video:=YOUR_VIDEO_OUTPUT_FILE_PATH.mp4
```

##### With live camera
```
roslaunch robot inference_nomad.launch 
```
Or, to record the video:   
```
roslaunch robot inference_nomad.launch record_video:=YOUR_VIDEO_OUTPUT_FILE_PATH.mp4
```
  
  


# Dataset

Milrem AIRE dataset processing scripts

## Default values for the arguments
- Default arguments are passed from the launch file. Please refer to the `launch` directory to see the actual argument values if needed.


## Installation, Creating Workspace and Building the Workspace
1. Create workspace
```
mkdir -p milrem_aire_ws/src   
cd milrem_aire_ws/src   
```

2. Clone the repo
```
git clone https://gitlab.cs.ut.ee/milrem/dataset.git . 
```

3. Install system dependencies
```
rosdep update   
rosdep install --from-paths . --ignore-src -r -y   
```

4. Build the workspace
```
cd ..   
catkin build   
```

5. Source the workspace   
```
source devel/setup.bash   
```

As this needs to be run every time before launching the nodes/executbles, it might be better to add it to the ~/.bashrc   

In the bashrc:   
```
source ~/milrem_aire_ws/devel/setup.bash   
```


## Extraction script
extracting csv files and images
- For a sequence of bags continuously (in those cases where the bags are not rerecorded for the same event/day)   
`python transform_projection/scripts/extraction_entire_event.py --bags_basename /home/anish/bags/2023-05-30*.bag`


- When the bags are re-recorded for the same event/day   
`python extraction_selective.py`   
In this case, the bags need to be selected manually (since there is no fixed range and each case is unique)   
Here, the name of the extraction (ouput folder) is set manually to the basename of the first bag in the list      
```
output_dir = os.path.join(args.output_directory, '2023-06-08-19-18-03')
os.makedirs(output_dir, exist_ok=True)

bag_files= ["/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-18-03_0.bag",
            "/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-19-04_1.bag",
            "/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-20-03_2.bag",
            "/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-21-03_3.bag",
            ]
```

## Visualization of predictions

### Conda environment setup
Please refer to the same conda environment as:   
`https://gitlab.cs.ut.ee/milrem/waypoint_planner/-/tree/master/#environment`

Please make sure that you install ros as well, as instucted in section 3 from `https://gitlab.cs.ut.ee/milrem/waypoint_planner/-/tree/master/#environment`

### Running the ros node
Firstly, activate the conda environment:   
```
conda activate milrem-aire
```

Inside the src directory of the workspace, clone the git repository:   
```
cd milrem_aire_ws/src   
git clone https://gitlab.cs.ut.ee/milrem/dataset.git .
```

Then, please make sure that you are in the root of the catkin workspace:   
```
cd milrem_aire_ws
```

Build all packages, and source the workspace:   
```
catkin build
source devel/setup.bash
```

From the root of the catkin workspace, run the following command:

```
rosrun waypoints_sampling trajectory_sampling.py --model_path <path where the model checkpoint is located> --model_type <type of the trained model> --waypoint_image_path <path of the waypoint image or goal image>
```
For example:   
```
rosrun waypoints_sampling trajectory_sampling.py --model_path /home/anish/milrem_aire_ws/src/waypoints_sampling/src/checkpoints/vae89.ckpt --model_type vae --waypoint_image_path /home/anish/waypoint_planner/img1683131892025955469.jpg
```

For validation purposes, the images are retrieved playing ros bags where the left camera images published by ZED 2i camera are subscribed   
```rosbag play --pause --clock <bagfile path>```
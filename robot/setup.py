from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    scripts=['nodes/inference_crops.py',
             'nodes/inference_nomad.py',             
             'nodes/image_publisher_node.py',
             'nodes/deadman_switch.py',
             'nodes/inference_distance_spaced_goals.py',             
             'nodes/dummy_onnx_run.py'
             ],
    packages=['waypoint_planner', 'global_planner', 'gnm_train', 'helpers', 'utils'],
    package_dir={'': 'src'},
    requires= ['rospy']
)

setup(**setup_args)

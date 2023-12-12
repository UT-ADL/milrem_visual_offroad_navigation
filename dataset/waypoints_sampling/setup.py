from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    scripts=['src/waypoints_sampling/nodes/trajectory_sampling.py', 'src/waypoints_sampling/nodes/visualize_frame.py'],
    packages=['waypoints_sampling'],
    package_dir={'': 'src'},
    requires= ['rospy']
)

setup(**setup_args)
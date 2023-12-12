import numpy as np
import os
import glob
import csv
import rosbag
import argparse
from pyproj import CRS, Transformer


def transform(lat, lon):

    coords = transformer.transform(lat, lon)

    easting = coords[0]
    northing = coords[1]

    return easting, northing

def extract_from_bag(output_dir, bag_files, topics):
    
    image_directory = output_dir + '/images'
    

    csv_directory =  output_dir + '/csv'
    

    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(csv_directory, exist_ok=True)

    csv_filepath = os.path.join(csv_directory, 'extracted_data.csv')

    pose_topic, image_topic, gnss_position_topic,gnss_orientation_topic, gnss_velocity_topic = topics
    
    
    with open(csv_filepath, 'a') as file:
        writer = csv.writer(file)
        header = ['image_name',
                    'timestamp',
                    'camera_position_x', 'camera_position_y', 'camera_position_z', 
                    'camera_orientation_x', 'camera_orientation_y', 'camera_orientation_z', 'camera_orientation_w',
                    'gnss_latitude', 'gnss_longitude', 'gnss_altitude',
                    'gnss_utm_easting', 'gnss_utm_northing',
                    'gnss_orientation_x', 'gnss_orientation_y', 'gnss_orientation_z', 'gnss_orientation_w',
                    'gnss_velocity_x', 'gnss_velocity_y', 'gnss_velocity_z']
                
        writer.writerow(header)
        
        pose_msg = None
        gnss_position_msg = None
        gnss_orientation_msg = None
        gnss_velocity_msg = None
        image_msg = None    

        for bag_file in bag_files:

            bag = rosbag.Bag(bag_file, 'r')

            for topic, msg, t in bag.read_messages():

                if topic == pose_topic:
                    pose_msg = msg.pose
                    position_x, position_y, position_z = pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z                     
                    orientation_x, orientation_y, orientation_z, orientation_w = pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w

                if topic == gnss_position_topic:
                    gnss_position_msg = msg
                    latitude, longitude, altitude = gnss_position_msg.vector.x, gnss_position_msg.vector.y, gnss_position_msg.vector.z 
                    utm_easting, utm_northing = transform(latitude, longitude)

                if topic == gnss_orientation_topic:
                    gnss_orientation_msg = msg
                    gnss_orientation_x, gnss_orientation_y, gnss_orientation_z, gnss_orientation_w = gnss_orientation_msg.quaternion.x, gnss_orientation_msg.quaternion.y, gnss_orientation_msg.quaternion.z, gnss_orientation_msg.quaternion.w 
                    

                if topic == gnss_velocity_topic:
                    gnss_velocity_msg = msg
                    velocity_x, velocity_y, velocity_z = gnss_velocity_msg.vector.x, gnss_velocity_msg.vector.y, gnss_velocity_msg.vector.z

                if topic == image_topic:
                    image_msg = msg
                    #image_name = 'img' + str(image_msg.header.stamp) + '.jpg'
                    image_name = 'img' + str(image_msg.header.seq) + '.jpg'
                    image_filepath = os.path.join(image_directory, image_name)
                    
                    timestamp = image_msg.header.stamp
                    
                    
                    with open(image_filepath, 'wb') as img:
                        img.write(image_msg.data)
                        
                    #if (pose_msg is not None) and (gnss_position_msg is not None) and (gnss_velocity_msg is not None):
                    if (pose_msg is None):
                        position_x = np.NaN
                        position_y = np.NaN
                        position_z = np.NaN
                        orientation_x = np.NaN
                        orientation_y = np.NaN
                        orientation_z = np.NaN
                        orientation_w = np.NaN
                    
                    if (gnss_position_msg is None):
                        latitude = np.NaN
                        longitude = np.NaN 
                        altitude = np.NaN
                        utm_easting = np.NaN
                        utm_northing = np.NaN
                    
                    if (gnss_orientation_msg is None):
                        gnss_orientation_x = np.NaN
                        gnss_orientation_y = np.NaN
                        gnss_orientation_z = np.NaN
                        gnss_orientation_w = np.NaN
                        
                    if (gnss_velocity_msg is None):
                        velocity_x = np.NaN
                        velocity_y = np.NaN
                        velocity_z = np.NaN
                        
                    writer.writerow([image_name,
                                    timestamp,
                                    position_x, position_y, position_z,
                                    orientation_x, orientation_y, orientation_z, orientation_w,
                                    latitude, longitude, altitude,
                                    utm_easting, utm_northing,
                                    gnss_orientation_x, gnss_orientation_y, gnss_orientation_z, gnss_orientation_w,
                                    velocity_x, velocity_y, velocity_z])
                    
                    print(f"Image {image_name} written to {image_directory}")
                    print(f"Added new row to {csv_filepath}")
                    print("--------------------------------") 

    
parser = argparse.ArgumentParser()

parser.add_argument("--pose_topic", type=str, help="Pose topic name", default='/zed2i/zed_node/odom')
parser.add_argument("--image_topic", type=str, help="Raw Image topic name", default='/zed2i/zed_node/left_raw/image_raw_color/compressed')
parser.add_argument("--gnss_position_topic", type=str, help="GNSS position topic name", default='/filter/positionlla')
parser.add_argument("--gnss_orientation_topic", type=str, help="GNSS orientation topic name", default='/filter/quaternion')
parser.add_argument("--gnss_velocity_topic", type=str, help="GNSS velocity topic name", default='/filter/velocity')
parser.add_argument("--output_directory", type=str, help="output directory name", default='/gpfs/space/projects/Milrem/extraction/extracted_datasets')
parser.add_argument("--bags_basename", type=str, help="bags base filepath")

args = parser.parse_args()

pose_topic = args.pose_topic
image_topic = args.image_topic
gnss_position_topic = args.gnss_position_topic
gnss_orientation_topic = args.gnss_orientation_topic
gnss_velocity_topic = args.gnss_velocity_topic

topics = [pose_topic, image_topic, gnss_position_topic, gnss_orientation_topic, gnss_velocity_topic]
print(f"Pose topic: {pose_topic}")
print(f"Image topic: {image_topic}")
print(f"GNSS position topic: {gnss_position_topic}")
print(f"GNSS orientation topic: {gnss_orientation_topic}")
print(f"GNSS velocity topic: {gnss_velocity_topic}")


crs_wgs84 = CRS.from_epsg(4326)
crs_utm = CRS.from_epsg(32635)
transformer = Transformer.from_crs(crs_wgs84, crs_utm)

#output_dir = args.output_directory + '/' + args.bags_basename.split('/')[-1]
#output_dir = output_dir.split('*')[0] + '_0'
# output_dir = args.output_directory + '/' + '2023-05-11_9'

# name of the ouput directory for extracted files
output_dir = os.path.join(args.output_directory, '2023-05-0319-07-25(3)')
os.makedirs(output_dir, exist_ok=True)

path_to_bags = args.bags_basename

bag_files = glob.glob(path_to_bags)

#bag_files= ["/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-18-03_0.bag",
            #"/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-19-04_1.bag",
            #"/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-20-03_2.bag",
            #"/gpfs/space/projects/Milrem/bagfiles/2023-06-08-19-21-03_3.bag",
            #]

extract_from_bag(output_dir=output_dir,
                 bag_files=bag_files,
                 topics=topics)


    

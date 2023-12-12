import numpy as np
import os
import glob
import rosbag
import argparse
import csv

def extract_from_bag(output_dir, bag_files, topics):

    img_dir = os.path.join(output_dir, 'images')
    csv_dir = os.path.join(output_dir, 'csv')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    csv_filepath = os.path.join(csv_dir, 'extracted_data.csv')
    odom_topic, image_topic = topics

    with open(csv_filepath, 'a') as file:
        writer = csv.writer(file)
        header = ['image_name',
                  'timestamp',
                  'position_x', 'position_y']
        writer.writerow(header)

        image_msg = None
        odom_msg = None

        for bag_file in bag_files:
            bag = rosbag.Bag(bag_file, 'r')

            for topic, msg, t in bag.read_messages():

                if topic == odom_topic:
                    odom_msg = msg
                    position_x, position_y = odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y

                if topic == image_topic:
                    image_msg = msg
                    timestamp = msg.header.stamp            
                    image_name = 'img' + str(image_msg.header.seq) + '.jpg'
                    # image_name = 'img' + str(timestamp) + '.jpg'
                    image_filepath = os.path.join(img_dir, image_name)

                    with open(image_filepath, 'wb') as img:
                        img.write(image_msg.data)
                    
                    if odom_msg is None:
                        position_x = np.NaN
                        position_y = np.NaN
                    
                    writer.writerow([image_name,
                                     timestamp,
                                     position_x, position_y])
                    
                    print(f"Image {image_name} written to {output_dir}")
                    print(f"New row added to {csv_filepath}")
                    print("-------------------------------------")

                
            
        bag.close()
        

parser = argparse.ArgumentParser()

parser.add_argument("--image_topic", type=str, default='/zed/zed_node/left_raw/image_raw_color/compressed')
parser.add_argument("--odom_topic", type=str, default='/odometry/filtered')
parser.add_argument("--output_dir", type=str, default='/home/adl/milrem_aire_ws/extracted_dataset')
parser.add_argument("--bags_basename", type=str)

args = parser.parse_args()

image_topic = args.image_topic
odom_topic = args.odom_topic

output_dir = args.output_dir

bag_files = glob.glob(args.bags_basename)

output_dir = os.path.join(output_dir, args.bags_basename.split('/')[-1])
output_dir = output_dir.split('*')[0]
output_dir = output_dir.split('.')[0]

os.makedirs(output_dir, exist_ok=True)

topics = [odom_topic, image_topic]

extract_from_bag(output_dir, bag_files, topics)

print("Extraction Complete !!")

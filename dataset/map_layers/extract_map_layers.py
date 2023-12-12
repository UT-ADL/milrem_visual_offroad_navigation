import rasterio
import os
import io
import requests
import pyproj
import subprocess
from owslib.wms import WebMapService
from PIL import Image
import numpy as np
from rasterio.warp import reproject, transform


class LandBoardWMSExtractor:

    def __init__(self, wms_service, output_folder, extract_existing_layers):
        
        wms_url = wms_service['url']
        self.wms = WebMapService(wms_url)

        self.wms_layer = wms_service['layer']
        self.srs = wms_service['srs']
        self.output_folder = output_folder
        self.appendix = wms_service['appendix']
        self.extract_existing_layers = extract_existing_layers


    def extract_and_save_orthophoto_from_land_board_wms(self, orienteering_maps):

        print("extracting %s for %i maps" % (self.appendix, len(orienteering_maps)))
        files = os.listdir(self.output_folder)

        # loop over keys and use enumerate to get the index
        for i, key in enumerate(orienteering_maps):

            print("   %i. %s" % (i+1, key), end="")
            file = key + "_" + self.appendix + '.tif'

            if self.extract_existing_layers == False and file in files:
                print(" - already exists")
                continue

            request = self.wms.getmap(
                layers=[self.wms_layer],
                styles='',
                srs=self.srs,
                format='image/jpeg',
                bbox=orienteering_maps[key]['bounds'],
                size=(orienteering_maps[key]['new_width'], orienteering_maps[key]['new_height'])
                )

            img_bytes = request.read()
            stream = io.BytesIO(img_bytes)
            image = Image.open(stream)
            r, g, b = image.split()

            transform = rasterio.transform.from_bounds(west=orienteering_maps[key]['bounds'].left,
                                                    south=orienteering_maps[key]['bounds'].bottom,
                                                    east=orienteering_maps[key]['bounds'].right,
                                                    north=orienteering_maps[key]['bounds'].top,
                                                    width=orienteering_maps[key]['new_width'],
                                                    height=orienteering_maps[key]['new_height'])
            crs = rasterio.crs.CRS.from_string(self.srs)
            # for some reasons dosn't work as expected
            # 
            compress_options = {'compress': 'lzw', 'zlevel': 9}
            out_file = os.path.join(self.output_folder, file)

            with rasterio.open(out_file,
                                'w', 
                                driver='GTiff',
                                compress=compress_options,
                                width=orienteering_maps[key]['new_width'], 
                                height=orienteering_maps[key]['new_height'], 
                                count=3, 
                                dtype='uint8', 
                                crs=crs, 
                                transform=transform
                                ) as dst:
                dst.write(np.array(r), 1)
                dst.write(np.array(g), 2)
                dst.write(np.array(b), 3)

            # compressed_file = os.path.join(self.output_folder + "/compressed/" + file)
            # # compress tiff
            # command = [
            #     'gdal_translate',
            #     '-of', 'GTiff',
            #     '-co', 'COMPRESS=LZW',
            #     '-co', 'BIGTIFF=YES',
            #     out_file,
            #     compressed_file
            # ]

            # subprocess.run(command, check=True)
            
            print(" - done")


class GoogleMapsExtractor:

    def __init__(self, google_layer, output_folder, extract_existing_layers, reduce_bbox_by_meters):
        self.output_folder = output_folder
        self.extract_existing_layers = extract_existing_layers

        self.maptype = google_layer['maptype']
        self.google_api_key = google_layer['key']
        self.zoom = google_layer['zoom']
        self.output_folder = output_folder
        self.appendix = google_layer['appendix']
        self.reduduce_bbox_by_meters = reduce_bbox_by_meters

        # create coordinate transformer from lest97 to wgs84
        src_crs = pyproj.CRS.from_string('EPSG:3301')
        dst_crs = pyproj.CRS.from_string('EPSG:4326')

        self.lest97_to_wgs84 = pyproj.Transformer.from_crs(src_crs, dst_crs)


    def extract_and_save_google_maps_data(self, orienteering_maps):

        print("extracting %s for %i maps" % (self.appendix, len(orienteering_maps)))

        files = os.listdir(self.output_folder)

        i = 0
        for key in orienteering_maps:
            i += 1

            print("   %i. %s" % (i, key), end="")

            file = key + "_" + self.appendix + '.jpg'

            if self.extract_existing_layers == False and file in files:
                print(" - already exists")
                continue

            bl_lat, bl_lon = self.lest97_to_wgs84.transform(orienteering_maps[key]['bounds'].bottom+self.reduduce_bbox_by_meters, orienteering_maps[key]['bounds'].left+self.reduduce_bbox_by_meters)
            br_lat, br_lon = self.lest97_to_wgs84.transform(orienteering_maps[key]['bounds'].bottom+self.reduduce_bbox_by_meters, orienteering_maps[key]['bounds'].right-self.reduduce_bbox_by_meters)
            tl_lat, tl_lon = self.lest97_to_wgs84.transform(orienteering_maps[key]['bounds'].top-self.reduduce_bbox_by_meters, orienteering_maps[key]['bounds'].left+self.reduduce_bbox_by_meters)
            tr_lat, tr_lon = self.lest97_to_wgs84.transform(orienteering_maps[key]['bounds'].top-self.reduduce_bbox_by_meters, orienteering_maps[key]['bounds'].right-self.reduduce_bbox_by_meters)
            # put together the bounding box string
            bbox = str(bl_lat) + "," + str(bl_lon) + "|" + str(tl_lat) + "," + str(tl_lon) + "|" + str(tr_lat) + "," + str(tr_lon) + "|" + str(br_lat) + "," + str(br_lon) + "|" + str(bl_lat) + "," + str(bl_lon)

            # calculate center coordinate and transform to wgs84
            center_northing = (orienteering_maps[key]['bounds'].top + orienteering_maps[key]['bounds'].bottom) / 2
            center_easting = (orienteering_maps[key]['bounds'].left + orienteering_maps[key]['bounds'].right) / 2
            c_lat, c_lon = self.lest97_to_wgs84.transform(center_northing, center_easting)

            url = 'https://maps.googleapis.com/maps/api/staticmap'
            params = {
                'center': str(c_lat) + "," + str(c_lon),  # Leave center parameter empty when using bounds
                'format': 'jpg',  # Specify image format
                'zoom': self.zoom,  # Specify the zoom level
                'size': '640x640',  # Specify the desired size of the image
                'maptype': self.maptype,  # Specify the map type
                'scale': 2,
                'path': 'color:0xff0000ff|weight:1|' + bbox,  # Draw a line on the map
                'key': self.google_api_key  # Specify your Google Maps API key
            }

            # Send a GET request to the API
            response = requests.get(url, params=params)

            out_file = os.path.join(self.output_folder, file)

            # Check if the request was successful
            if response.status_code == 200:
                # Save the response content to a file
                with open(out_file, 'wb') as f:
                    f.write(response.content)
                    print(' - done')
            else:
                print('Error:', response.status_code)



def transform_gtiff_to_UTM(in_raster, out_raster):

    # https://epsg.io/25835 - ETRS89 / UTM zone 35N
    dst_crs = pyproj.CRS.from_string('EPSG:25835')

    # Open the source raster file
    with rasterio.open(in_raster) as src:
        # Retrieve the metadata and CRS of the source raster
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height

        # Define the transformation parameters for the target CRS
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(src_crs, dst_crs, src_width, src_height, *src.bounds)

        # Create a new raster file with the target CRS
        with rasterio.open(out_raster, 'w', driver='GTiff', count=src.count, dtype='uint8', crs=dst_crs, transform=dst_transform, width=dst_width, height=dst_height) as dst:
            for band_idx in range(1, src.count + 1):
                # Read the band from the source raster
                band_data = src.read(band_idx)

                # Reproject the band to the target CRS
                reproject(
                    source=band_data,
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.enums.Resampling.nearest  # Adjust the resampling method if needed
                )
        
        dst.close()

    print(out_raster + " transformed")


def extract_orienteering_map_data(folder):

    dict = {}

    files = os.listdir(folder)
    files = [f for f in files if f.endswith('.tif')]

    # extract attributs into dictionary
    for file in files:
        file_path = os.path.join(folder, file)
        with rasterio.open(file_path) as src:

            scale = 1.0

            # remove the extension from file name
            file_name = os.path.splitext(file)[0]
            dict[file_name] = {}

            dict[file_name]['bounds'] = src.bounds
            dict[file_name]['crs'] = src.crs
            # recalculate the width and height of the image to have the same resolution
            dict[file_name]['orig_width'] = src.width
            dict[file_name]['orig_height'] = src.height
            dict[file_name]['orig_res'] = src.transform[0]
            
            new_width = int((src.bounds.right - src.bounds.left)/src.transform[0])
            new_height = int((src.bounds.top - src.bounds.bottom)/src.transform[0])
            
            if new_width > 4096:
                scale = 4096 / new_width
            
            if new_height > 4096:
                scale = min(scale, 4096 / new_height)

            dict[file_name]['new_width'] = min(4096, int(new_width * scale))
            dict[file_name]['new_height'] = min(4096, int(new_height * scale))
            dict[file_name]['new_res'] = src.transform[0] / scale

            if scale < 1.0:
                print("adjusted file: %s, with scale: %f" % (file_name, scale))

    return dict
import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self, fill=False):
        # fill object area with depth color
        self.fill = fill
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
    	# Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None
        
        if self.fill:
            # Apply filter to fill the Holes in the depth image
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.holes_fill, 3)
            filtered_depth = spatial.process(depth_frame)
            
            hole_filling = rs.hole_filling_filter()
            filled_depth = hole_filling.process(filtered_depth)
            
            # Create colormap to show the depth of the Objects
            colorizer = rs.colorizer()
            depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())
            
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()

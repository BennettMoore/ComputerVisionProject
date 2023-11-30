import csv
import numpy as np
import open3d as o3d
import os

# A set of distinct colors to distinguish different clusters
CLUSTER_COLORS = np.array([
    [1.,0.,0.], # Red
    [0.,1.,0.], # Green
    [0.,0.,1.], # Blue
    [1.,1.,0.], # Yellow
    [1.,0.,1.], # Magenta
    [0.,1.,1.], # Cyan 
    [0.5,1.,0.], # Lime
    [0.5,0.,1.], # Purple
    [1.,0.5,0.], # Orange
    [0.,0.5,1.], # Sky Blue
    [0.,1.,0.75], # Turquoise
    [1.,0.5,0.75], # Pink
    [0.5,0.5,1.], # Lavender
    [0.75,0.75,0.75], # Gray
])
NUM_CARS = 6
STARTING_INDEX = 0
ENDING_INDEX = 499
STARTING_ID = 141 # He used this in the CSV
FILE_HEADER = ['vehicle_id','position_x','position_y','position_z','mvec_x','mvec_y','mvec_z','bbox_x_min','bbox_x_max','bbox_y_min','bbox_y_max','bbox_z_min','bbox_z_max']
HEIGHT_THRESHOLD = 4
MIN_POINTS = 2
CLUSTERING_EPSILON = 2
ERROR_MARGIN = 1.
DIR_PATH = os.path.join(os.getcwd(), 'dataset', 'PointClouds')


def background_subtract(curr, prev):
    min_distance = 0.001
    if prev is not None:
        distances = curr.compute_point_cloud_distance(prev)
        distances = np.asarray(distances)
        ind = np.where(distances > min_distance)[0]
        return curr.select_by_index(ind)
    return curr

def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    prev = None
    stable_flag = False
    prev_midpoints = None
    for i in range(STARTING_INDEX, ENDING_INDEX + 1):
        file_path = os.path.join(DIR_PATH, f'{i}.pcd')
        pointcloud = o3d.io.read_point_cloud(file_path)
        # First handle background subtraction
        result = background_subtract(pointcloud, prev)
        # Then, use height thresholding to get rid of any points not belonging to cars
        points = np.asarray(result.points)
        filtered_points = points[points[:, 2] <= HEIGHT_THRESHOLD]
        
        prev = pointcloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        labels = np.array(filtered_pcd.cluster_dbscan(CLUSTERING_EPSILON, MIN_POINTS))
        max_label = labels.max()
        
        cluster_bboxes = []
        cluster_pcds = []
        # print(labels)
        
        # Debug statement to view number of clusters
        print(f"Frame {i} has {max_label + 1} clusters")

        clusters = {}
        midpoints = {}
        mvecs = {}
        bboxes = {}
        # Include this for now, so that we start processing when we have the first stable frame (if he grades it with another set of pointclouds)
        if max_label + 1 == NUM_CARS and stable_flag is False:
            stable_flag = True
            with open(f'perception_results/frame_{i}.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(FILE_HEADER)
                for j in range(NUM_CARS-1, -1, -1):
                    this_cluster = []
                    for k in range(len(labels)):
                        if j == labels[k]:
                            this_cluster.append([filtered_points[k][0], filtered_points[k][1], filtered_points[k][2]])
                    num_points = len(this_cluster)
                    curr_id = STARTING_ID + j
                    
                    x_sum = sum(point[0] for point in this_cluster)
                    y_sum = sum(point[1] for point in this_cluster)
                    z_sum = sum(point[2] for point in this_cluster)
                    x_min = min(point[0] for point in this_cluster)
                    y_min = min(point[1] for point in this_cluster)
                    z_min = min(point[2] for point in this_cluster)
                    x_max = max(point[0] for point in this_cluster)
                    y_max = max(point[1] for point in this_cluster)
                    z_max = max(point[2] for point in this_cluster)
                    
                    # Create colored point cloud and bounding box to display later
                    cluster_cloud = o3d.geometry.PointCloud()
                    cluster_cloud.points = o3d.utility.Vector3dVector(this_cluster)
                    cluster_cloud.paint_uniform_color(CLUSTER_COLORS[j])
                    
                    cluster_bbox = cluster_cloud.get_axis_aligned_bounding_box()
                    cluster_bbox.color = CLUSTER_COLORS[j]
                    
                    cluster_pcds.append(cluster_cloud)
                    cluster_bboxes.append(cluster_bbox)
                    clusters[curr_id] = this_cluster

                    midpoint = np.asarray((x_sum / num_points, y_sum / num_points, z_sum / num_points))
                    bbox = np.asarray((midpoint[0] - x_min, x_max - midpoint[0],midpoint[1] - y_min, y_max - midpoint[1], midpoint[2] - z_min, z_max - midpoint[2]))
                    midpoint = np.asarray((x_sum / num_points, y_sum / num_points, z_sum / num_points))
                    bbox = np.asarray((midpoint[0] - x_min, x_max - midpoint[0],midpoint[1] - y_min, y_max - midpoint[1], midpoint[2] - z_min, z_max - midpoint[2]))
                    #print(bbox_z_max)
                    #print(bbox_z_min)
                    
                    csv_writer.writerow([curr_id, midpoint[0], midpoint[1], midpoint[2], 0., 0., 0., bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]])
                    midpoints[curr_id] = midpoint
                    mvecs[curr_id] = np.asarray((0., 0., 0.))
                    bboxes[curr_id] = bbox
                prev_midpoints = midpoints
                prev_mvecs = mvecs
                prev_bboxes = bboxes
                prev_clusters = clusters
            for j in range(STARTING_INDEX, i):
                with open(f'perception_results/frame_{i}.csv', 'r', newline='') as csv_file:
                    with open(f'perception_results/frame_{j}.csv', 'w', newline='') as old_csv_file:
                        reader = csv.reader(csv_file)
                        prev_data = list(reader)
                        writer = csv.writer(old_csv_file)
                        writer.writerows(prev_data)

        if stable_flag:
            with open(f'perception_results/frame_{i}.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(FILE_HEADER)
                
                raw_midpoints = []
                raw_bboxes = []
                raw_clusters = []
                for j in range(max_label+1):
                    this_cluster = []
                    for k in range(len(labels)):
                        if j == labels[k]:
                            this_cluster.append([filtered_points[k][0], filtered_points[k][1], filtered_points[k][2]])
                    num_points = len(this_cluster)
                    
                    if num_points != 0: # If cluster is empty, ignore it
                        x_sum = sum(point[0] for point in this_cluster)
                        y_sum = sum(point[1] for point in this_cluster)
                        z_sum = sum(point[2] for point in this_cluster)
                        x_min = min(point[0] for point in this_cluster)
                        y_min = min(point[1] for point in this_cluster)
                        z_min = min(point[2] for point in this_cluster)
                        x_max = max(point[0] for point in this_cluster)
                        y_max = max(point[1] for point in this_cluster)
                        z_max = max(point[2] for point in this_cluster)
                        
                        raw_midpoint = np.asarray((x_sum / num_points, y_sum / num_points, z_sum / num_points))
                        raw_midpoints.append(np.asarray((x_sum / num_points, y_sum / num_points, z_sum / num_points)))
                        raw_bboxes.append(np.asarray((raw_midpoint[0] - x_min, x_max - raw_midpoint[0],raw_midpoint[1] - y_min, y_max - raw_midpoint[1], raw_midpoint[2] - z_min, z_max - raw_midpoint[2])))
                        raw_clusters.append(this_cluster)
                        #print(bbox_z_max)
                        #print(bbox_z_min)
                
                # Use previous frame's positions to estimate where each car should be
                for est_id, est in prev_midpoints.items():
                    if len(raw_midpoint) == 0 or len(raw_bboxes) == 0: # No clusters left, so just estimate
                        midpoint = est
                        mvec = prev_mvecs[est_id]
                        bbox = prev_bboxes[est_id]
                        cluster_data = prev_clusters[est_id]
                    else:
                        bbox_diffs = np.asarray([np.linalg.norm(r - prev_bboxes[est_id]) for r in raw_bboxes])
                        min_bbox = np.argmin(bbox_diffs)
                        
                        # Use Bounding Box dimensions to identify each car since they remain fairly constant
                        if bbox_diffs[min_bbox] <= ERROR_MARGIN:
                            # print(f"Suitable match for Vehicle {est_id} found! {bbox_diffs[min_bbox]} <= {ERROR_MARGIN}")
                            # print(f"\tBounding box diffs: {bbox_diffs}")
                            midpoint = raw_midpoints[min_bbox]
                            mvec = midpoint - prev_midpoints[est_id]
                            bbox = raw_bboxes[min_bbox]
                            cluster_data = raw_clusters[min_bbox]

                            # Once a cluster is chosen, remove it from the pool to prevent duplicates
                            raw_midpoints.pop(min_bbox)
                            raw_bboxes.pop(min_bbox)
                            raw_clusters.pop(min_bbox)
                        else: # If no clusters match, just estimate where the car should be
                            # print(f"No suitable match for Vehicle {est_id} found. {bbox_diffs[min_bbox]} > {ERROR_MARGIN}")
                            # print(f"\tBounding box diffs: {bbox_diffs}")
                            midpoint = est
                            mvec = prev_mvecs[est_id]
                            bbox = prev_bboxes[est_id]
                            cluster_data = prev_clusters[est_id]
                            
                    # Create colored point cloud and bounding box to display later
                    cluster_cloud = o3d.geometry.PointCloud()
                    cluster_cloud.points = o3d.utility.Vector3dVector(cluster_data)
                    cluster_cloud.paint_uniform_color(CLUSTER_COLORS[est_id - STARTING_ID])
                    
                    cluster_bbox = cluster_cloud.get_axis_aligned_bounding_box()
                    cluster_bbox.color = CLUSTER_COLORS[est_id - STARTING_ID]
                    
                    cluster_pcds.append(cluster_cloud)
                    cluster_bboxes.append(cluster_bbox)
                    
                    csv_writer.writerow([est_id, midpoint[0], midpoint[1], midpoint[2], mvec[0], mvec[1], mvec[1], bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]])
                    midpoints[est_id] = midpoint
                    mvecs[est_id] = mvec
                    bboxes[est_id] = bbox
                    clusters[est_id] = cluster_data
                    # This is where we would try thge algo I talked about 
                    
                prev_midpoints = midpoints
                prev_mvecs = mvecs
                prev_bboxes = bboxes
                prev_clusters = clusters
                    
            
        # o3d.visualization.draw_geometries([filtered_pcd])
        vis.clear_geometries()
        for cloud in cluster_pcds:
            vis.add_geometry(cloud)
        for bbox in cluster_bboxes:
            vis.add_geometry(bbox)
        vis.poll_events()
        vis.update_renderer()
        
        # Clean up point clouds once they're rendered
        cluster_pcds.clear()
        cluster_bboxes.clear()
    
if __name__ == '__main__':
    main()
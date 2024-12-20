import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def visualize_poses_open3d(poses):
    """
    Visualize a large number of 7D poses (position + quaternion) in 3D using Open3D.

    Args:
        poses: NumPy array of shape (N, 7), where each row is [x, y, z, qx, qy, qz, qw].
    """
    vis_objects = []  # List to store Open3D objects

    for pose in poses:
        # Extract position and quaternion
        position = pose[:3]
        quaternion = pose[3:]

        # Create a small sphere to represent the position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(position)  # Move sphere to the pose's position
        sphere.paint_uniform_color([0, 0, 1])  # Blue color
        vis_objects.append(sphere)
        
        # draw the camera normal vector
        forward_vector = np.array([0, 0, 1])

        rotation = R.from_quat(quaternion)
        camera_direction = rotation.apply(forward_vector)



        camera_normal = position + camera_direction * 0.3
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([position, camera_normal])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0]])

        vis_objects.append(line)

        # Create a coordinate frame to represent the orientation
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=position)
        # # calculate rotation matrix from quaternion
        # rotation_matrix = R.from_quat(quaternion).as_matrix()
        # # rotate the frame
        # frame.rotate(rotation_matrix, center=position)


        # # Create a coordinate frame to represent the orientation
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=position)

        # # Apply the rotation defined by the quaternion
        # rotation_matrix = R.from_quat(quaternion).as_matrix()  # Convert quaternion to rotation matrix
        # frame.rotate(rotation_matrix, center=position)  # Rotate around the pose's position
        # vis_objects.append(frame)

    # Create a coordinate frame to represent the orientation
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    vis_objects.append(frame)

    # draw table as rectangle
    table = o3d.geometry.TriangleMesh.create_box(width=0.6, height=0.6, depth=0.01)
    table.translate([-0.3,-0.3,-0.005])
    table.paint_uniform_color([0.6, 0.6, 0])
    vis_objects.append(table)

    # Visualize all objects in Open3D
    o3d.visualization.draw_geometries(vis_objects)

# Example usage
poses = np.load("tcp_poses_test5.npy")  # Load poses from a file
poses[:, 3:] = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)  # Normalize quaternions
visualize_poses_open3d(poses)

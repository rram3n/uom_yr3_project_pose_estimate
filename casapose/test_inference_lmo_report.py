from __future__ import print_function
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import time
import math
import json

from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist
import trimesh

from casapose.utils.geometry_utils import apply_offsets, project
from casapose.utils.draw_utils import draw_bb, draw_axes
from casapose.data_handler.image_only_dataset import ImageOnlyDataset
from casapose.pose_estimation.pose_evaluation import poses_pnp
from casapose.pose_estimation.voting_layers_2d import CoordLSVotingWeighted
from casapose.pose_models.tfkeras import Classifiers


def main():
    """Main function to run the inference pipeline."""
    # Print transformed points in decimal notation
    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
    # Define paths
    MODEL_PATH = "./data/pretrained_models/result_w_8.h5"
    MODEL_NAME = "result_w_8.h5"
    OUTPUT_BASE_PATH = "./output_report/lmo_test"
    
    # IMAGE_PATH = "./DATAPATH/lmo_report_test/ten_img/rgb"
    # gt_path = "./DATAPATH/lmo_report_test/ten_img/gt"

    IMAGE_PATH = "./DATAPATH/lmo_report_test/full_img_set/rgb"
    gt_path = "./DATAPATH/lmo_report_test/full_img_set/gt"

    #os.makedirs(OUTPUT_PATH, exist_ok=True)
    camera = 'dataset_cam' # 'webcam', 'dataset_cam', 'blender_cam'

    desired_object = 'obj_000012'

    #     obj_dict = {
    #     "obj_000001": "ape", 
    #     "obj_000005": "wateringcan",
    #     "obj_000006": "cat", # Not good
    #     "obj_000008": "drill",
    #     "obj_000009": "duck",
    #     "obj_000010": "eggbox", # Not good
    #     "obj_000011": "glue",
    #     "obj_000012": "holepuncher"
    # }

    # Make incremental output path
    run_id = 0
    OUTPUT_PATH = f"{OUTPUT_BASE_PATH}_{run_id}"
    while os.path.exists(OUTPUT_PATH):
        if not os.listdir(OUTPUT_PATH):  # If folder is empty, reuse it
            abs_output_path = os.path.abspath(OUTPUT_PATH)
            print(f"Reusing existing empty folder: {OUTPUT_PATH}")
            print("output path = ", abs_output_path)
            break
        run_id += 1
        OUTPUT_PATH = f"{OUTPUT_BASE_PATH}_{run_id}"

    # Create the folder only if it does not exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        abs_output_path = os.path.abspath(OUTPUT_PATH)
        print("output path = ", abs_output_path)

    # Temporary folder to output images and model checkpoints, it will add a train_ in front of the name
    CHECKPOINT_PATH = OUTPUT_PATH + "tmp/training_checkpoints" 

    # Data configuration
    height, width = 480, 640  # Image dimensions

    if camera == 'webcam':
        camera_matrix = np.array([
            [[567.4848, 0, 355.3875],
            [0, 566.3179, 246.73947],
            [0, 0, 1]]])
    elif camera == 'dataset_cam':
        camera_matrix = np.array([
            [[572.4114, 0., 325.2611],
            [0., 573.57043, 242.049],
            [0., 0., 1.]]])
    elif camera == 'blender_cam':
        camera_matrix = np.array([
            [[888.88888889, 0, 320.0],
            [0, 666.66666667, 240.0],
            [0, 0, 1]]])
        
        
    # Obtain object list, dictionary, keypoint array and cuboid array from function
    print("Initializing object data...\n")
    obj_list, obj_dict, keypoints_array, cuboids_array = declare_objects()


    keypoints = tf.convert_to_tensor(keypoints_array, dtype=tf.float16)
    cuboids = tf.convert_to_tensor(cuboids_array, dtype=tf.float16)

    # print("keypoints: ", keypoints, "\n")
    # print("cuboids: ", cuboids, "\n")

    no_objects = len(obj_list) # 8
    print("number of objects: ",no_objects,"\n")

    no_points = 9
    #print(no_points)

    test_dataset = ImageOnlyDataset(root=IMAGE_PATH)
    #print(len(test_dataset))

    print("Object initilized!\n")


    print("Processing images...\n")
    testingdata, test_batches = process_images(IMAGE_PATH)
    testingdata_iterator = iter(testingdata)
    print("Image successfully processed!\n")


    print("Loading model...\n")
    net = load_model(MODEL_PATH, MODEL_NAME, CHECKPOINT_PATH, height, width, no_objects, no_points)
    print("Model successfully loaded!\n")


    print("Running inference... \n")
    test_batches = int(test_batches)

    loader_iterator, img_batches, est_poses, coords_list, conf_list = runnetwork(net, testingdata_iterator, test_batches, camera_matrix, keypoints, no_objects+1, no_points, OUTPUT_PATH)

    #print("Estimated poses: ", est_poses)
    #print("Confidence shape: ", tf.shape(conf_list))

    print("Inference completed! \n")

    ## Draw BB, centroid, and save poses in file
    print("Drawing bounding boxes...")
    image_output_path = os.path.join(OUTPUT_PATH, "images")  # Corrected path assignment
    
    cam_mat = camera_matrix[0]

    testingdata_iterator = iter(testingdata)

    desired_object_est_pose_arr = []

    # loops trhough all testing data
    for n, img_name in tqdm(enumerate(test_dataset.imgs), total=len(test_dataset.imgs)):
        img_batch = testingdata_iterator.get_next()
        ### IMAGEONLY
        img_batch = tf.expand_dims(img_batch, 0)
        img_batch = tuple(img_batch)

        img = img_batch[0][0]

        # EST POSE
        obj_idx = obj_list.index(desired_object)
        gt_poses = cuboids_array[obj_idx][0] # shape is (8,3)
        poses_est = est_poses[n] # shape is (1, 8, 1, 3, 4) which represents (batch size, no of detected objects, instances of an object?, rotation matrix, transformation matrix)
        
        # Find matching ground truth frame
        current_frame_str = f"{n:06d}.json"
        gt_pose_center = read_obj8_poses(gt_path, desired_object)
        matching_gt = None
        
        # Find matching ground truth pose
        for gt_pose in gt_pose_center:
            if gt_pose['filename'] == current_frame_str:
                matching_gt = gt_pose
                break
        
        if matching_gt is None:
            print(f"No matching ground truth found for frame {current_frame_str}")
            continue


        # print("GT pose file: ", matching_gt['filename'])
        # print("GT pose:\n", matching_gt['full_transform'], "\n")
        #print("Estimated poses shape: ", poses_est.shape, "\n")

        poses_est = np.reshape(poses_est, (1, 8, 3, 4))
        estimated_poses=poses_est[0] # shapes to (8, 3, 4)
        est_pose_conf_val = conf_list[n][0]

        gt_keypoints = keypoints_array[0][obj_idx][0][1:]

        # print("GT keypoints: ", gt_keypoints, "\n")
        # print("GT keypoints shape: ", gt_keypoints.shape, "\n")

        #print("estimated poses: ",estimated_poses, "\n")

        #print(img_name)

        file_prefix = img_name.split('/')[-1].split('.')[0]

        # Applies bounding boxes and saves images
        desired_object_est_pose, desried_object_est_keypoints = draw_bb_inference(
            desired_object,
            gt_poses,
            obj_dict,
            obj_list,
            n,
            img = img, 
            estimated_poses=estimated_poses,
            cuboids=cuboids,
            camera_matrix=cam_mat,
            path=image_output_path,
            file_prefix=file_prefix,
            gt_pose_info=matching_gt,
            confidence_scores=est_pose_conf_val
        )

        desired_object_est_pose_arr.append(desired_object_est_pose)

    print("Bounding box drawn!")

    print("Computing pose errors...")
    avg_trans_error, avg_rot_error, avg_rot_error_mat, avg_add_error = compute_pose_errors(
        desired_object_est_pose_arr,
        desried_object_est_keypoints,
        gt_path,
        OUTPUT_PATH,
        obj_list,
        desired_object,
        camera_matrix,
    )
    print(f"Average translation error: {avg_trans_error:.2f} mm")
    print(f"Average rotation error euler: {avg_rot_error:.2f} degrees")
    print(f"Average rotation error matrix: {avg_rot_error_mat:.2f} degrees")
    print(f"Average ADD error: {avg_add_error:.2f}\n")

def declare_objects():
    """Returns object list, dictionary, keypoints, and cuboids."""

    obj_list = [
        'obj_000001', 'obj_000005', 'obj_000006', 'obj_000008',
        'obj_000009', 'obj_000010', 'obj_000011', 'obj_000012'
    ]

    obj_dict = {
        "obj_000001": "ape",
        "obj_000005": "wateringcan",
        "obj_000006": "cat",
        "obj_000008": "drill",
        "obj_000009": "duck",
        "obj_000010": "eggbox",
        "obj_000011": "glue",
        "obj_000012": "holepuncher"
    }
    keypoints_array = np.array([[
                                #"obj_000001": "ape"
                                [[[-9.7732497e-03, 3.6659201e-03, -1.4534000e-03],
                                [3.6046398e+01, -1.4680500e+01, -4.5020599e+01],
                                [-3.0289101e+01, -7.2402501e+00, -4.2632900e+01],
                                [7.4425201e+00, 2.3966400e+01, -3.9362701e+01],
                                [-4.3485899e+00, 3.6769500e+00, 4.5835800e+01],
                                [6.2882501e-01, -3.6412800e+01, -2.7732599e+01],
                                [-2.6754901e-01, 3.7588799e+01, -4.7640200e+00],
                                [3.0029400e+01, -2.3939800e+01, -8.1097898e+00],
                                [-2.8789900e+01, -1.9449200e+01, -9.0417604e+00]]],

                                #"obj_000005": "wateringcan"
                                [[[1.6300200e-02, -2.3040799e-03, -1.1291500e-02],
                                [5.5248199e+00, 5.4157101e+01, -9.6322701e+01],
                                [-4.1018100e+00, 1.2732400e+01, 9.6678497e+01],
                                [-9.1580000e+00, -4.1244202e+01, -8.7472397e+01],
                                [7.3375401e+00, 9.0886101e+01, -1.1365300e+01],
                                [-1.0262200e+01, -9.0547600e+01, -3.7563899e-01],
                                [-4.7794201e+01, 1.6508699e+01, -5.6376900e+01],
                                [4.8287998e+01, 2.4022501e+00, -6.2877899e+01],
                                [4.6154099e+01, 1.1302400e+01, 4.9851101e+01]]],

                                # "obj_000006": "cat"
                                [[[1.7128000e-02, -4.5700101e-03, -5.3901700e-03],
                                [2.0947300e+01, -6.1587502e+01, -5.4198200e+01],
                                [-2.0933701e+01, 6.3563000e+01, 2.6130899e+01],
                                [2.8901501e+01, 2.7392700e+01, -5.7568199e+01],
                                [1.4403200e+00, -5.8665901e+01, 2.2473900e+01],
                                [1.2946500e+01, 1.4082400e+01, 5.8292999e+01],
                                [-2.8743299e+01, 1.6301001e+01, -5.2558300e+01],
                                [-3.3441200e+01, -4.1310501e+01, -5.4232101e+01],
                                [2.3869900e+01, 4.1699699e+01, 1.6587299e+01]]],

                                #"obj_000008": "drill"
                                [[[-2.4108901e-03, -6.2332200e-03, -6.3247699e-03],
                                [1.1291000e+02, -3.4727199e+00, 9.2172699e+01],
                                [-1.1182900e+02, 3.1709600e-02, 6.1154400e+01],
                                [-6.2377201e+01, 1.0970700e+01, -1.0025700e+02],
                                [4.2661201e+01, -2.4666700e+01, -9.9452499e+01],
                                [1.0724100e+01, -3.5357201e+00, 1.0133300e+02],
                                [-4.1970699e+01, -3.1155399e+01, 5.4645599e+01],
                                [4.9310899e+00, 3.6434399e+01, -9.7123596e+01],
                                [5.6840302e+01, -4.2665200e+00, 4.8058399e+01]]],

                                #"obj_000009": "duck"
                                [[[-3.4179699e-03, -9.8838797e-03, 3.9329501e-03],
                                [4.9320702e+01, 6.2302999e+00, -4.0302898e+01],
                                [-4.6246700e+01, 2.3396499e+00, -3.7502899e+01],
                                [1.2448000e+01, -3.3365299e+01, -4.0734501e+01],
                                [3.9640200e+00, 3.4297600e+01, -4.0923302e+01],
                                [4.5272598e+01, -1.0067500e+00, 2.1399401e+01],
                                [6.6833901e+00, -3.1548400e+00, 4.2783199e+01],
                                [-2.3509399e+01, -2.7834400e+01, -1.9335600e+01],
                                [-4.1355202e+01, 1.3988900e-01, 1.3391900e+00]]],

                                #"obj_000010": "eggbox"
                                [[[-1.7417900e-02, -4.2999300e-01, -1.3252300e-02],
                                [-7.0443398e+01, 4.3526299e+01, 4.2999201e+00],
                                [7.3233902e+01, 3.5586300e+01, 4.8644700e+00],
                                [6.7131897e+01, -4.4466202e+01, -2.7725799e+00],
                                [-7.0990898e+01, -3.6974701e+01, -1.3353300e+00],
                                [-4.7924999e+01, 5.5036702e+00, -3.2836399e+01],
                                [2.2584101e+01, 4.1242500e+01, 3.2724400e+01],
                                [-2.4753901e+01, -4.0470100e+01, 3.2213699e+01],
                                [4.7744598e+01, 4.2735401e-01, -3.1653799e+01]]],

                                #"obj_000011": "glue"
                                [[[9.9391900e-03, -1.1459400e-02, 6.8359398e-03],
                                [-9.1147299e+00, -3.1402399e+01, -8.5777802e+01],
                                [9.7676700e-01, 2.9348700e+00, 8.6390404e+01],
                                [6.4356799e+00, 3.7870701e+01, -6.3978802e+01],
                                [9.7071304e+00, -3.6640800e+01, -3.6885799e+01],
                                [-1.5302700e+01, 1.4431200e+00, -4.7971500e+01],
                                [-6.0784298e-01, -1.2160700e+01, 4.3689098e+01],
                                [1.7079800e+01, 1.9666600e+00, -8.3763802e+01],
                                [-4.1084499e+00, 3.5197800e+01, -2.3239799e+01]]],
                                
                                #"obj_000012": "holepuncher"
                                [[[-0.00317764, -0.00389862, 0.0116768],
                                [45.3747, 47.1416, -37.7327],
                                [-34.1938, -52.1613, -38.8932],
                                [-37.2968, 48.9418, -39.6015],
                                [39.7582, 48.9375, 35.2916],
                                [44.2335, -42.2343, -37.7516],
                                [39.6273, -44.8461, 35.1788],
                                [-15.6101, 52.0532, 11.4769],
                                [-50.3098, -12.8502, -5.48413]]]
                                
                                ]])

    cuboids_array = np.array([
                                #"obj_000001": "ape"
                                [[[-37.92094, -38.788555, -45.88129],
                                [-37.92094, -38.788555, 45.87838],
                                [-37.92094, 38.795883, -45.88129],
                                [-37.92094, 38.795883, 45.87838],
                                [37.901394, -38.788555, -45.88129],
                                [37.901394, -38.788555, 45.87838],
                                [37.901394, 38.795883, -45.88129],
                                [37.901394, 38.795883, 45.87838]]],

                                #"obj_000005": "wateringcan"
                                [[[-50.35713, -90.89071, -96.8516],
                                [-50.35713, -90.89071, 96.82902],
                                [-50.35713, 90.8861, -96.8516],
                                [-50.35713, 90.8861, 96.82902],
                                [50.38973, -90.89071, -96.8516],
                                [50.38973, -90.89071, 96.82902],
                                [50.38973, 90.8861, -96.8516],
                                [50.38973, 90.8861, 96.82902]]],

                                # "obj_000006": "cat"
                                [[[-33.44303, -63.791, -58.71809],
                                [-33.44303, -63.791, 58.707314],
                                [-33.44303, 63.781857, -58.71809],
                                [-33.44303, 63.781857, 58.707314],
                                [33.47729, -63.791, -58.71809],
                                [33.47729, -63.791, 58.707314],
                                [33.47729, 63.781857, -58.71809],
                                [33.47729, 63.781857, 58.707314]]],

                                #"obj_000008": "drill"
                                [[[-114.72308, -37.718895, -103.983604],
                                [-114.72308, -37.718895, 103.97095],
                                [-114.72308, 37.706425, -103.983604],
                                [-114.72308, 37.706425, 103.97095],
                                [114.71827, -37.718895, -103.983604],
                                [114.71827, -37.718895, 103.97095],
                                [114.71827, 37.706425, -103.983604],
                                [114.71827, 37.706425, 103.97095]]],

                                #"obj_000009": "duck"
                                [[[-52.200897, -38.71081, -42.8214],
                                [-52.200897, -38.71081, 42.82927],
                                [-52.200897, 38.691044, -42.8214],
                                [-52.200897, 38.691044, 42.82927],
                                [52.194057, -38.71081, -42.8214],
                                [52.194057, -38.71081, 42.82927],
                                [52.194057, 38.691044, -42.8214],
                                [52.194057, 38.691044, 42.82927]]],

                                #"obj_000010": "eggbox"
                                [[[-75.0917, -54.39756, -34.629425],
                                [-75.0917, -54.39756, 34.602924],
                                [-75.0917, 53.53758, -34.629425],
                                [-75.0917, 53.53758, 34.602924],
                                [75.05686, -54.39756, -34.629425],
                                [75.05686, -54.39756, 34.602924],
                                [75.05686, 53.53758, -34.629425],
                                [75.05686, 53.53758, 34.602924]]],

                                #"obj_000011": "glue"
                                [[[-18.320473, -38.923126, -86.376724],
                                [-18.320473, -38.923126, 86.3904],
                                [-18.320473, 38.900208, -86.376724],
                                [-18.320473, 38.900208, 86.3904],
                                [18.340351, -38.923126, -86.376724],
                                [18.340351, -38.923126, 86.3904],
                                [18.340351, 38.900208, -86.376724],
                                [18.340351, 38.900208, 86.3904]]],

                                #"obj_000012": "holepuncher"
                                [[[-50.4439, -54.2485, -45.4],
                                [-50.4439, -54.2485, 45.4],
                                [-50.4439, 54.2485, -45.4],
                                [-50.4439, 54.2485, 45.4],
                                [50.4440, -54.2485, -45.4],
                                [50.4440, -54.2485, 45.4],
                                [50.4440, 54.2485, -45.4],
                                [50.4440, 54.2485, 45.4]]]
                                
                                ])
    
    return obj_list, obj_dict, keypoints_array, cuboids_array

def read_obj8_poses(folder_path, desired_object):
    """
    Read 6D poses of object 8 from JSON files in the given folder.
    
    Args:
        folder_path: Path to folder containing JSON files
    
    Returns:
        list of dict: Each containing location and orientation for object 8
    """
    poses = []
    
    # List and sort all JSON files
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Find object 8 in the objects list
            for obj in data['objects']:
                if obj['class'] == desired_object:
                    # Get rotation and translation
                    pose_transform = np.array(obj['pose_transform'])
                    rotation = pose_transform[:3, :3]
                    translation = np.array(obj['location'])
                    keypoints_3d = np.array(obj['keypoints_3d'])[1:]
                    
                    # Construct correct transformation matrix
                    correct_transform = np.eye(4)
                    correct_transform[:3, :3] = rotation
                    correct_transform[:3, 3] = translation
                    
                    pose_info = {
                        'filename': filename,
                        'object':obj['class'],
                        'location': translation,  # [x, y, z] in mm
                        'rotation_matrix': rotation.tolist(),  # 3x3 rotation matrix
                        'full_transform': correct_transform,  # 4x4 transformation matrix
                        'keypoints_3d': keypoints_3d  # [9, 3] keypoints in camera frame
                    }
                    poses.append(pose_info)
                    break
    
    return poses

def load_model(model_path, model_name, checkpoint_path, height, width, no_objects, no_points):
    """Loads the CASAPose model with pretrained weights."""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in path: {model_path}")
    else:
        abs_model_path = os.path.abspath(model_path)
        print("Model path found: ", abs_model_path)


    #CASAPose = Classifiers.get("casapose_cond_weighted") # casapose_cond_weighted from original test code
    CASAPose = Classifiers.get("casapose_c_gcu5") # casapose_c_gcu5 from Berkeley code
    ver_dim = no_points * 2
    ver_dim += no_points # from estimate_confidence = True
        

    net = CASAPose(
        ver_dim=ver_dim,
        seg_dim=1 + no_objects,
        input_shape=(height, width, 3),
        input_segmentation_shape=None,
        weights="imagenet",
        base_model="resnet18"
    )

    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
    checkpoint = tf.train.Checkpoint(network=net)  # , optimizer=optimizer)

    checkpoint = tf.train.Checkpoint(network=net)
    """ 
    net_path = "./tmp/frozen_model/data/pretrained_models/result_w_8.h5"
    print(net_path)
    net.load_weights(net_path, by_name=True, skip_mismatch=True)
    """
    net.load_weights(model_path, by_name=True, skip_mismatch=True)

    for layer in net.layers:
        layer.trainable = False

    #net.summary()

    return net

def process_images(image_path):
    """Loads and processes images from the dataset."""
    test_dataset = ImageOnlyDataset(root=image_path)
    testingdata, test_batches = test_dataset.generate_dataset(1)
    return testingdata, test_batches

def runnetwork(net, loader_iterator, batches, camera_matrix, keypoints, no_objects, no_points, output_path):
    """Runs the inference on the dataset using the trained model."""
    inference_times = []
    total_inference_time = 0
    fps_list = []

    @tf.function
    def test_step(img_batch):
        start_time = tf.timestamp()
        net_input = [tf.expand_dims(img_batch[0][0], 0)]
        output_net = net(net_input, training=False)  # all stages are present here
        output_seg, output_dirs, confidence = tf.split(output_net, [no_objects, no_points * 2, -1], 3)

        coordLSV_in = [output_seg, output_dirs, confidence]
        coords = CoordLSVotingWeighted(
            name="coords_ls_voting",
            num_classes=no_objects,
            num_points=no_points,
            filter_estimates=True,
        )(coordLSV_in)

        poses_est = poses_pnp(
            coords, output_seg, keypoints, camera_matrix, no_objects-1, min_num=200
        )

        seg_confidence = tf.reduce_mean(output_seg, axis=[1, 2])  # Average over spatial dimensions
        # Get confidence from the confidence branch
        point_confidence = tf.reduce_mean(confidence, axis=[1, 2])  # Average confidence per object
        # Combine both confidences
        object_confidence = seg_confidence * point_confidence  # Element-wise multiplication
        # Remove background class
        object_confidence = object_confidence[:, 1:]  # [1, no_objects-1]

        end_time = tf.timestamp()
        time_needed = end_time - start_time
        return time_needed, poses_est, coords, object_confidence
    
    speed = []
    img_batches = []
    est_poses = []
    coords_list = []
    conf_list = []

    # Create timing output file
    timing_file = os.path.join(output_path, "inference_timing.txt")
    with open(timing_file, "w") as f:
        # Write table header
        f.write("| {:^6} | {:^9} | {:^7} |\n".format("Frame", "Time (ms)", "FPS"))
        f.write("|{:=^8}|{:=^11}|{:=^9}|\n".format("", "", ""))

    # Runs inference batch-by-batch
    for batch_idx in tqdm(range(batches)):
        img_batch = loader_iterator.get_next()
        ### IMAGEONLY
        img_batch = tf.expand_dims(img_batch, 0)
        img_batch = tuple(img_batch)
        
        # Estimates 6D poses for each object
        time_needed, poses_est, coords, conf_val = test_step(img_batch)
        
        inference_time = (time_needed) * 1000  # Convert to milliseconds
        fps = 1.0 / (time_needed)
        
        # Store timing information
        inference_times.append(inference_time)
        fps_list.append(fps)

        # Log timing information
        with open(timing_file, "a") as f:
            f.write("| {:6d} | {:9.2f} | {:7.2f} |\n".format(batch_idx, inference_time, fps))

        img_batches.append(img_batch)
        est_poses.append(poses_est)
        coords_list.append(coords)
        conf_list.append(conf_val)

    # Calculate and print statistics
    total_inference_time = sum(inference_times[1:])
    avg_inference_time = total_inference_time / batches
    avg_fps = len(inference_times) / (total_inference_time / 1000)
    
    print("\nInference Speed Statistics:")
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Min FPS: {min(fps_list):.2f}")
    print(f"Max FPS: {max(fps_list):.2f}")
    
    # Save summary statistics
    with open(os.path.join(output_path, "inference_timing.txt"), "a") as f:
        f.write("\n")
        f.write("=" * 50 + "\n")
        f.write("Inference Speed Summary:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total frames processed: {batches}\n")
        f.write(f"Total inference time: {total_inference_time/1000:.2f} seconds\n")
        f.write(f"Average inference time: {avg_inference_time:.2f} ms\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write(f"Min FPS: {min(fps_list):.2f}\n")
        f.write(f"Max FPS: {max(fps_list):.2f}\n")

    return loader_iterator, img_batches, est_poses, coords_list, conf_list

def project_relative(xyz, K, RT):
    """
    Project points relative to the estimated position.
    
    xyz: [N, 3] Points relative to estimated position
    K: [3, 3] Camera matrix
    RT: [3, 4] Transformation matrix
    """
    # Get the estimated position (translation part of RT)
    est_position = RT[:, 3]
    
    # Convert relative coordinates to absolute coordinates
    xyz_abs = xyz + est_position
    
    # Project using standard projection
    xyz = np.dot(xyz_abs, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, xyz_abs

def draw_axes_correct(img, keypoints, colors = [(255, 0, 0), (0, 255, 0),(0, 125, 255)], thickness = 2 ):
    # args: image, projected_cuboid keypoints, list of 3 colors to use, tickenss
    # returns the image with the line drawn
    
    # finds the center point
    center = np.mean(keypoints, axis=0)
    center = [int(i) for i in center]
    
    # finds the top of the object    
    point1_top = [(keypoints[1][0] + keypoints[2][0])/2, (keypoints[1][1] + keypoints[2][1])/2]
    point2_top = [(keypoints[0][0] + keypoints[3][0])/2, (keypoints[0][1] + keypoints[3][1])/2]
    top_coords = [int((point1_top[0] + point2_top[0])/2), int((point1_top[1] + point2_top[1])/2)]
    
    
    # finds the right of the top of the object
    point1_right = [(keypoints[3][0] + keypoints[6][0])/2, (keypoints[3][1] + keypoints[6][1])/2]
    point2_right = [(keypoints[2][0] + keypoints[7][0])/2, (keypoints[2][1] + keypoints[7][1])/2]
    right_coords = [int((point1_right[0] + point2_right[0])/2), int((point1_right[1] + point2_right[1])/2)]
    
    # finds the center of the front of the object
    point1_front = [(keypoints[1][0] + keypoints[7][0])/2, (keypoints[1][1] + keypoints[7][1])/2]
    point2_front = [(keypoints[3][0] + keypoints[5][0])/2, (keypoints[3][1] + keypoints[5][1])/2]
    front_coords = [int((point1_front[0] + point2_front[0])/2), int((point1_front[1] + point2_front[1])/2)]
    
    # draws lines
    cv2.line(img, center, top_coords, colors[0], thickness)
    cv2.putText(img, 'X', top_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)

    cv2.line(img, center, right_coords, colors[1], thickness)
    cv2.putText(img, 'Y', right_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 2)

    cv2.line(img, center, front_coords, colors[2], thickness)
    cv2.putText(img, 'Z', front_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 2)

    return img

def draw_bb_inference(
    desired_object,
    gt_poses, # the cuboid points in the ground truth pose [1,8,3]
    obj_dict,
    obj_list,
    n,
    img, # [480, 640, 3]
    estimated_poses, # [8, 3, 4]
    cuboids, # [8, 1, 8, 3]
    camera_matrix, # [1, 3, 3]
    path,
    file_prefix,
    gt_pose_info=None, # center ground truth pose [4x4]
    normal=[0.5, 0.5],
    confidence_scores=None):

    """
    Draws bounding boxes and keypoints on images and saves them
    """
    gt_pose = gt_pose_info['full_transform']
    desired_object_index = obj_list.index(desired_object)
    desried_object_est_pose = []
    desried_object_est_keypoints = []

    colors = {'blue': (255, 0, 0), 'red': (220,20,60), 'white': (255, 255, 255)}
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    # Create the output directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Create a separate directory for coordinate files
    coord_dir = os.path.join(path, "coordinates")
    os.makedirs(coord_dir, exist_ok=True)
    
    # Define coordinate output path
    coord_output_path = os.path.join(coord_dir, f"img_{n}_coordinates.txt")
    
    current_frame_str = f"{n:06d}.json"
    gt_filename = gt_pose_info['filename']

    if gt_filename != current_frame_str:
        print(f'Frame number mismatch: Expected {current_frame_str}, got {gt_filename}')
        return desried_object_est_pose, desried_object_est_keypoints

    with open(coord_output_path, "w") as file:
        file.write("Object Detection Coordinates\n")
        file.write("=" * 50 + "\n\n")
        
    #  Converts normalized image (0-1 range) back to 8-bit (0-255) and creates a copy (img_cuboids) to avoid modifying the original image
    img_keypoints = tf.cast(((img * normal[1]) + normal[0]) * 255, dtype=tf.uint8).numpy()
    img_cuboids = img_keypoints.copy()

    # Ensures that poses with near values are skipped
    eps = 1e-4
    
    # Reshapes pose matrix into [8, 3, 4] → [Objects, Rotation + Translation] e.g. 3x3 rotation, 3x1 translation
    estimated_poses = np.reshape(estimated_poses, (8, 3, 4))

    # Extracts pose and cuboid for each object
    for obj_idx, obj_pose in enumerate(estimated_poses):
        inst_idx = 0
        obj_pose_est = estimated_poses[obj_idx]
        instance_cuboids = cuboids[obj_idx][inst_idx]
        
        # Draw BB (white bounding box represents the estimated pose)
        valid_est = np.abs(np.sum(obj_pose_est)) > eps # True if pose valid e.g. not near zero
        if valid_est: 
            transformed_cuboid_points2d, transformed_points3d = project(instance_cuboids, camera_matrix, obj_pose_est)   
            transformed_cuboid_points2d = np.reshape(transformed_cuboid_points2d, (8,2))
            transformed_points3d = np.reshape(transformed_points3d, (8,3))

            # Retrieve the object name from the dictionary
            text = obj_dict[obj_list[obj_idx]]
            thickness = 2
            color = colors['white']

            # Get center for coordinate logging
            center = np.mean(transformed_cuboid_points2d, axis=0)
            center = [int(i) for i in center]

            # Draw bounding box only if object is drill
            if obj_idx == desired_object_index:
                #print("Estimated Pose: \n", obj_pose_est)
                desried_object_est_pose = obj_pose_est
                desried_object_est_keypoints.append(transformed_points3d)

                draw_bb(transformed_cuboid_points2d, img_cuboids, (255, 255, 255))
                #draw_axes(img=img_cuboids, keypoints=transformed_cuboid_points2d)
                draw_axes_correct(img=img_cuboids, keypoints=transformed_cuboid_points2d)

                # Find the top-most point of the bounding box
                top_y = np.min(transformed_cuboid_points2d[:, 1])
                center_x = np.mean(transformed_cuboid_points2d[:, 0])

                # Position the label above the bounding box
                label_position = (int(center_x), int(top_y - 10))  # 10 pixels above the top

                # Label the object at the center of its bounding box
                img_test2 = cv2.putText(img_cuboids, 
                                    text, 
                                    label_position, #[50, 100], 
                                    fontFace = font, 
                                    fontScale = fontScale, 
                                    color = color, 
                                    thickness = thickness)
                
                # Label the estimated pose center with a small marker
                marker_color = (255, 255, 0)  # Yellow marker for visibility
                img_cuboids = cv2.circle(
                    img_cuboids, 
                    tuple(center),  # Marker at center
                    radius=4, 
                    color=marker_color, 
                    thickness=-1  # Fill the circle
                )

                # Ground Truth (GT) --> if ground truth is avaliable, draw blue BB
                if gt_pose is not None: 
                    gt_pose_R = gt_pose[:3, :3].T  # Transpose rotation matrix
                    gt_pose_t = gt_pose[:3, 3]
                    
                    # Construct the correct 3x4 matrix
                    gt_pose_reshaped = np.zeros((3, 4))
                    gt_pose_reshaped[:3, :3] = gt_pose_R
                    gt_pose_reshaped[:3, 3] = gt_pose_t
                    valid_gt = np.abs(np.sum(gt_pose_reshaped)) > eps
                    if valid_gt:
                        # print("\nGT Poses: \n",gt_poses)
                        # print("GT Pose shape: \n",gt_poses.shape)
                        print(f"\nGT Pose for frame {gt_pose_info['filename']}\n")
                        print(f"Current frame: {current_frame_str}")

                        transformed_cuboid_points2d_gt, transformed_points3d_gt = project(instance_cuboids,camera_matrix, gt_pose_reshaped)
                        # print("\nInstance Cuboids:\n",instance_cuboids)

                        # print("\nBB Transformed Cuboid Points 3D Est:\n", transformed_points3d)
                        # print("BB Transformed Cuboid Points 3D GT:\n", transformed_points3d_gt)

                        draw_bb(transformed_cuboid_points2d_gt, img_cuboids, (0, 0, 255),width=1)
                        #draw_axes_correct(img=img_cuboids, keypoints=transformed_cuboid_points2d_gt)
                        # Find the top-most point of the bounding box
                        bottom_y = np.max(transformed_cuboid_points2d_gt[:, 1])
                        center_x = np.mean(transformed_cuboid_points2d_gt[:, 0])

                        # Position the label below the bounding box
                        label_position = (int(center_x), int(bottom_y + 15))
                        text_label = "GT"
                        img_test3 = cv2.putText(img_cuboids, 
                                    text_label, 
                                    label_position, #[50, 100], 
                                    fontFace = font, 
                                    fontScale = fontScale, 
                                    color = (0, 0, 255), 
                                    thickness = thickness)
                else: 
                    print('invaid gt pose')
                    continue

            # Write to file
            obj_name = obj_dict[obj_list[obj_idx]]
            with open(coord_output_path, "a") as file:  # Changed to append mode
                file.write(f"Object {obj_list[obj_idx]}: {obj_name}\n")

                if confidence_scores is not None:
                    conf = confidence_scores[obj_idx]
                    file.write(f"Confidence Score: {conf:.4f}\n")

                file.write(f"2D Center coordinates (x,y): {center}\n")
                file.write(f"Full pose matrix (in camera frame):\n{obj_pose_est}\n")
                
                # Add keypoints to output
                keypoints_2d, _ = project(instance_cuboids, camera_matrix, obj_pose_est)
                keypoints_2d = np.reshape(keypoints_2d, (8,2))
                file.write("\nKeypoints (2D image coordinates):\n")
                for i, kp in enumerate(keypoints_2d):
                    file.write(f"Keypoint {i+1}: ({int(kp[0])}, {int(kp[1])})\n")
                
                # Add 3D keypoints in camera frame
                file.write("\nKeypoints (3D camera frame):\n")
                for i, kp in enumerate(instance_cuboids):
                    file.write(f"Keypoint {i+1}: ({kp[0]:.2f}, {kp[1]:.2f}, {kp[2]:.2f})\n")
                
                file.write("-" * 50 + "\n\n")
        else: 
            # print(obj_idx)
            # print('skipped obj est')
            continue

    # Save image
    os.makedirs(path, exist_ok=True)
    img_cuboids = Image.fromarray((img_cuboids).astype("uint8"))
    #img_cuboids.save(path + "/" + str(file_prefix) + "_cuboids_all.png")
    img_cuboids.save(path + "/" + str(n) + ".png")
    
    '''
    # Uses matplotlib to display the image
    #img2 = img_test2[:,:,::-1]
    img2 = img_cuboids
    plt.figure(figsize=(10*2,6*2))
    plt.imshow(img2)
    cv2.destroyAllWindows()
    '''
    return desried_object_est_pose, desried_object_est_keypoints
    
def compute_pose_errors(est_poses, est_keypoints, gt_folder_path, output_path, obj_list, desired_object, K):
    """
    Compare estimated poses with ground truth and compute errors.
    
    Args:
        est_poses: List of estimated poses from inference
        gt_folder_path: Path to ground truth JSON files
        output_path: Path to save error analysis
        obj_list: List of object IDs
    """
    # Get ground truth poses
    gt_poses = read_obj8_poses(gt_folder_path, desired_object)
    obj_idx = obj_list.index(desired_object)
    _, _, keypoints_array, cuboids_array = declare_objects()
    model_points = keypoints_array[0][obj_idx][0]  # Skip center point, use 8 keypoints
    model_points_cuboid = cuboids_array[obj_idx][0]

    # mesh_path = os.path.join(f"./DATAPATH/lmo/models/{desired_object}/{desired_object}.ply")
    # print("Mesh path: ", mesh_path)
    # mesh = trimesh.load(mesh_path)
    # mesh_points = np.array(mesh.vertices)
    # model_diameter = np.max(pdist(mesh_points))

    model_diameter = np.max(pdist(model_points_cuboid))
    threshold = 0.2 * model_diameter  # 10% of diameter

    print("\nModel Diameter: ", model_diameter)
    print("Threshold: ", threshold)

    translation_errors = []
    rotation_errors = []
    rotation_errors_mat =[]
    failed_frames = []
    add_errors = []
    correct_pose_count = 0
    total_poses = 0
    
    # Create error log file
    error_file = os.path.join(output_path, "pose_errors.txt")
    with open(error_file, "w") as f:
        f.write(f"Pose Errors for {desired_object}\n")
        f.write("=" * 50 + "\n\n")
        
        # Process each frame
        for frame_idx, (est_pose, gt_pose_info) in enumerate(zip(est_poses, gt_poses)):
            try:
                # Get estimated pose components
                est_R = est_pose[:3, :3]
                est_t = est_pose[:3, 3]

                
                # Get ground truth pose components
                gt_t = gt_pose_info['location']
                gt_R = np.array(gt_pose_info['rotation_matrix']).T

                gt_mat = np.zeros((3, 4))
                gt_mat[:3, :3] = gt_R
                gt_mat[:3, 3] = gt_t

                # Compute translation error vector and magnitude (in mm)
                trans_error_vector = est_t - gt_t
                trans_error = np.linalg.norm(trans_error_vector)  # Magnitude
                
                # Convert rotation matrices to Euler angles (in degrees)
                est_euler = Rotation.from_matrix(est_R).as_euler('xyz', degrees=True)
                gt_euler = Rotation.from_matrix(gt_R).as_euler('xyz', degrees=True)

                # Compute rotation errors for each axis
                rot_error_vector = est_euler - gt_euler
                rot_error = np.linalg.norm(rot_error_vector)  # Total rotation error

                # Calculate rotation error using matrix logarithm (better than Euler angles)
                R_diff = est_R @ gt_R.T
                trace_val = np.trace(R_diff)
                theta_rad = np.arccos(np.clip((trace_val - 1) / 2.0, -1.0, 1.0))
                # Convert the angle to degrees:
                rot_error_mat = np.degrees(theta_rad)
                
                # Calculate ADD error
                transformed_est = np.dot(model_points_cuboid, est_R.T) + est_t
                transformed_gt = np.dot(model_points_cuboid, gt_R.T) + gt_t

                # print("\nError Transformed Transation Est:\n", transformed_est)
                # print("Error Transformed Transation GT:\n", transformed_gt)

                # _, transformed_est = project(model_points_cuboid, K, est_pose)
                # _, transformed_gt = project(model_points_cuboid, K, gt_mat)

                # Calculate point-wise distances
                distances = np.linalg.norm(transformed_est - transformed_gt, axis=1)
                add_error = np.mean(distances)
                add_errors.append(add_error)
            
                # print("Model Point\n",model_points_cuboid)
                # print("\nError Transformed Transation Est:\n", transformed_est)
                # print("Error Transformed Transation GT:\n", transformed_gt)

                # print('GT Rotation:\n', gt_R)
                # print('Est Rotation:\n', est_R)
                # print('R_diff:\n', R_diff)
                # print('rot_error_mat:\n', rot_error_mat)

                is_correct_pose = np.floor(add_error) <= np.ceil(threshold)

                if is_correct_pose:
                    correct_pose_count += 1
                total_poses += 1
                
                # Store errors
                translation_errors.append(trans_error)
                rotation_errors.append(rot_error)
                rotation_errors_mat.append(rot_error_mat)

                # Write frame results
                f.write(f"Frame {frame_idx} ({gt_pose_info['filename']}):\n")
                f.write("\nEstimated Pose:\n")
                f.write("Rotation:\n")
                f.write(f"{est_R[0]}\n{est_R[1]}\n{est_R[2]}\n")
                f.write(f"Roll (X): {est_euler[0]:.2f}\n")
                f.write(f"Pitch (Y): {est_euler[1]:.2f}\n")
                f.write(f"Yaw (Z): {est_euler[2]:.2f}\n")
                f.write("Translation: ")
                f.write(f"[{est_t[0]:.2f}, {est_t[1]:.2f}, {est_t[2]:.2f}]\n")

                f.write("\nGround Truth Pose:\n")
                f.write("Rotation:\n")
                f.write(f"{gt_R[0]}\n{gt_R[1]}\n{gt_R[2]}\n")
                f.write("Euler Angles (xyz, degrees):\n")
                f.write(f"Roll (X): {gt_euler[0]:.2f}\n")
                f.write(f"Pitch (Y): {gt_euler[1]:.2f}\n")
                f.write(f"Yaw (Z): {gt_euler[2]:.2f}\n")
                f.write("Translation: ")
                f.write(f"[{gt_t[0]:.2f}, {gt_t[1]:.2f}, {gt_t[2]:.2f}]\n")

                # Write error analysis
                f.write("\nError Analysis:\n")
                f.write("Rotation Error (degrees):\n")
                f.write(f"Roll (X): {rot_error_vector[0]:.2f}\n")
                f.write(f"Pitch (Y): {rot_error_vector[1]:.2f}\n")
                f.write(f"Yaw (Z): {rot_error_vector[2]:.2f}\n")
                f.write(f"Total Rotation Error Euler: {rot_error:.2f}\n")
                f.write(f"Total Rotation Error Matrix: {rot_error_mat:.2f}\n")
                f.write("Translation Error Vector [x, y, z]: ")
                f.write(f"[{trans_error_vector[0]:.2f}, {trans_error_vector[1]:.2f}, {trans_error_vector[2]:.2f}] mm\n")
                f.write(f"Translation Error Magnitude: {trans_error:.2f} mm\n")

                f.write("\nADD Error Analysis:\n")
                f.write(f"ADD Error: {add_error:.2f}\n")
                f.write(f"Point-wise distances: {distances}\n")
                f.write(f"Model diameter: {model_diameter:.2f} mm\n")
                f.write(f"10% threshold: {threshold:.2f} mm\n")
                f.write(f"Pose correct: {is_correct_pose}\n")
                f.write(f"Percentage of model diameter: {add_error/model_diameter*100:.2f}\n")
                
                f.write("-" * 30 + "\n")

                print(f"Frame {frame_idx} ADD Error: {add_error}")
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                failed_frames.append(frame_idx)
                continue
        
        if translation_errors:
            # Compute statistics
            avg_trans_error = np.mean(translation_errors)
            avg_rot_error = np.mean(rotation_errors)
            avg_rot_error_mat = np.mean(rotation_errors_mat)
            
            # Standard deviation
            std_trans_error = np.std(translation_errors)
            std_rot_error = np.std(rotation_errors)
            std_rot_error_mat = np.std(rotation_errors_mat)
            avg_add_error = np.mean(add_errors)
            std_add_error = np.std(add_errors)
            
            # Write summary statistics
            f.write("\nSummary Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of frames processed: {len(translation_errors)}\n")
            f.write(f"Number of failed frames: {len(failed_frames)}\n")
            f.write(f"Failed frames: {failed_frames}\n")
            f.write(f"Average translation error: {avg_trans_error:.2f} +- {std_trans_error:.2f} mm\n")
            f.write(f"Average rotation error euler: {avg_rot_error:.2f} +- {std_rot_error:.2f} degrees\n")
            f.write(f"Average rotation error mat: {avg_rot_error_mat:.2f} +- {std_rot_error_mat:.2f} degrees\n")
            f.write(f"Average ADD error: {avg_add_error:.2f} +- {std_add_error:.2f}\n")
            f.write(f"Number of correct poses: {correct_pose_count}\n")
            f.write(f"Number of ADD errors >= 100: {np.sum(np.array(add_errors) >= 100)}\n")
            f.write(f"ADD error array: {add_errors}\n")

            print("\nPose Error Statistics:")
            print(f"Number of frames processed: {len(translation_errors)}\n")
            print(f"Number of failed frames: {len(failed_frames)}\n")
            print(f"Number of correct poses: {correct_pose_count}")
            print(f"Number of ADD errors >= 100: {np.sum(np.array(add_errors) >= 100)}")

            return avg_trans_error, avg_rot_error, avg_rot_error_mat, avg_add_error
        else:
            print("No valid poses found for comparison")
            return float('nan'), float('nan')
        
if __name__ == "__main__":
    main()

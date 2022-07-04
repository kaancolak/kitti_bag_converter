import os
from datetime import datetime

import numpy as np

import rosbag2_py

from rclpy.serialization import serialize_message
# from std_msgs.msg import String

from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField, Imu, NavSatFix, NavSatStatus, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Point, Quaternion, Twist, TwistWithCovariance, TwistWithCovarianceStamped, Pose, \
    PoseStamped, TransformStamped, TwistStamped, Transform

from autoware_auto_perception_msgs.msg import DetectedObjects, DetectedObject, ObjectClassification, Shape

from tf2_msgs.msg import TFMessage
from rosgraph_msgs.msg import Clock
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy

import pykitti
import tf_transformations

import sys
from sys import argv as cmdLineArgs
from xml.etree.ElementTree import ElementTree
from warnings import warn

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0': STATE_UNSET, '1': STATE_INTERP, '2': STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1': OCC_UNSET, '0': OCC_VISIBLE, '1': OCC_PARTLY, '2': OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {'99': TRUNC_UNSET, '0': TRUNC_IN_IMAGE, '1': TRUNC_TRUNCATED, '2': TRUNC_OUT_IMAGE,
                 '3': TRUNC_BEHIND_IMAGE}


class Tracklet(object):
    objectType = None
    size = None  # len-3 float array: (height, width, length)
    firstFrame = None
    trans = None  # n x 3 float array (x,y,z)
    rots = None  # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None  # n x 2 uint8 array  (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation
    amtOccs = None  # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
    amtBorders = None  # None (n x 3) float array  (amt_border_l / _r / _kf)
    nFrames = None

    def __init__(self):
        self.size = np.nan * np.ones(3, dtype=float)

    def __str__(self):
        return '[Tracklet over {0} frames for {1}]'.format(self.nFrames, self.objectType)

    def __iter__(self):
        if self.amtOccs is None:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                       itertools.repeat(None), itertools.repeat(None),
                       range(self.firstFrame, self.firstFrame + self.nFrames))
        else:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                       self.amtOccs, self.amtBorders, range(self.firstFrame, self.firstFrame + self.nFrames))


def parseXML(trackletFile):
    # convert tracklet XML data to a tree structure
    eTree = ElementTree()
    print('Parsing tracklet file', trackletFile)
    with open(trackletFile) as f:
        eTree.parse(f)

    # now convert output to list of Tracklet objects
    trackletsElem = eTree.find('tracklets')
    tracklets = []
    trackletIdx = 0
    nTracklets = None
    for trackletElem in trackletsElem:
        # print 'track:', trackletElem.tag
        if trackletElem.tag == 'count':
            nTracklets = int(trackletElem.text)
            print('File contains', nTracklets, 'tracklets')
        elif trackletElem.tag == 'item_version':
            pass
        elif trackletElem.tag == 'item':
            newTrack = Tracklet()
            isFinished = False
            hasAmt = False
            frameIdx = None
            for info in trackletElem:
                # print 'trackInfo:', info.tag
                if isFinished:
                    raise ValueError('more info on element after finished!')
                if info.tag == 'objectType':
                    newTrack.objectType = info.text
                elif info.tag == 'h':
                    newTrack.size[0] = float(info.text)
                elif info.tag == 'w':
                    newTrack.size[1] = float(info.text)
                elif info.tag == 'l':
                    newTrack.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    newTrack.firstFrame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        # print 'trackInfoPose:', pose.tag
                        if pose.tag == 'count':  # this should come before the others
                            if newTrack.nFrames is not None:
                                raise ValueError('there are several pose lists for a single track!')
                            elif frameIdx is not None:
                                raise ValueError('?!')
                            newTrack.nFrames = int(pose.text)
                            newTrack.trans = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.rots = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.states = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.occs = np.nan * np.ones((newTrack.nFrames, 2), dtype='uint8')
                            newTrack.truncs = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.amtOccs = np.nan * np.ones((newTrack.nFrames, 2), dtype=float)
                            newTrack.amtBorders = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            frameIdx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frameIdx is None:
                                raise ValueError('pose item came before number of poses!')
                            for poseInfo in pose:
                                # print 'trackInfoPoseInfo:', poseInfo.tag
                                if poseInfo.tag == 'tx':
                                    newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    newTrack.states[frameIdx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    newTrack.occs[frameIdx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    newTrack.occs[frameIdx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    newTrack.truncs[frameIdx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    newTrack.amtOccs[frameIdx, 0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    newTrack.amtOccs[frameIdx, 1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    newTrack.amtBorders[frameIdx, 0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    newTrack.amtBorders[frameIdx, 1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    newTrack.amtBorders[frameIdx, 2] = float(poseInfo.text)
                                    hasAmt = True
                                else:
                                    raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
                            frameIdx += 1
                        else:
                            raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    isFinished = True
                else:
                    raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))
            # end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not isFinished:
                warn('tracklet {0} was not finished!'.format(trackletIdx))
            if newTrack.nFrames is None:
                warn('tracklet {0} contains no information!'.format(trackletIdx))
            elif frameIdx != newTrack.nFrames:
                warn('tracklet {0} is supposed to have {1} frames, but perser found {1}!'.format(trackletIdx,
                                                                                                 newTrack.nFrames,
                                                                                                 frameIdx))
            if np.abs(newTrack.rots[:, :2]).sum() > 1e-16:
                warn('track contains rotation other than yaw!')

            if not hasAmt:
                newTrack.amtOccs = None
                newTrack.amtBorders = None

            # add new tracklet to list
            tracklets.append(newTrack)
            trackletIdx += 1

        else:
            raise ValueError('unexpected tracklet info')

    print('Loaded', trackletIdx, 'tracklets.')

    if trackletIdx != nTracklets:
        warn('according to xml information the file has {0} tracklets, but parser found {1}!'.format(nTracklets,
                                                                                                     trackletIdx))

    return tracklets


##################
##################
##################
##################


basedir = '/home/kaan/KITTI/kitti_0093/'
date = '2011_09_26'
drive = ['0093']

semimajor_axis = 6378137.0
semiminor_axis = 6356752.31424518
pi = 3.14159265359


def se3_translation(lat, lon, alt, scale):
    tx = scale * lon * pi * semimajor_axis / 180.0
    ty = scale * semimajor_axis * np.log(np.tan((90.0 + lat) * pi / 360.0))
    tz = alt
    return [tx, ty, tz]


def save_pcl(bag, kitti, velo_frame_id, topic, new_topic):
    print("Exporting velodyne data")
    if new_topic:
        velo_topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic,
            type='sensor_msgs/msg/PointCloud2',
            serialization_format='cdr')
        bag.create_topic(velo_topic_info)
    velo_path = os.path.join(kitti.data_path, 'velodyne_points')
    velo_data_dir = os.path.join(velo_path, 'data')
    velo_filenames = sorted(os.listdir(velo_data_dir))
    with open(os.path.join(velo_path, 'timestamps.txt')) as f:
        lines = f.readlines()
        velo_datetimes = []
        for line in lines:
            if len(line) == 1:
                continue
            dt = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            velo_datetimes.append(dt)

    iterable = zip(velo_datetimes, velo_filenames)
    for dt, filename in iterable:
        if dt is None:
            continue
        velo_filename = os.path.join(velo_data_dir, filename)

        scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)

        depth = np.linalg.norm(scan, 2, axis=1)
        pitch = np.arcsin(scan[:, 2] / depth)  # arcsin(z, depth)
        fov_down = -24.8 / 180.0 * np.pi
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        proj_y = (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        proj_y *= 64  # in [0.0, H]
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(64 - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y = proj_y.reshape(-1, 1)
        scan = np.concatenate((scan, proj_y), axis=1)
        scan = scan.tolist()
        for i in range(len(scan)):
            scan[i][-1] = int(scan[i][-1])

        t = float(datetime.strftime(dt, "%s.%f"))

        timer = Time(seconds=int(t), nanoseconds=int((t - int(t)) * 10 ** 9))
        header = Header(stamp=timer.to_msg(), frame_id=velo_frame_id)

        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                  PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1)]
        pcl_msg = point_cloud2.create_cloud(header, fields, scan)
        pcl_msg.is_dense = True
        bag.write(topic, serialize_message(pcl_msg), timer.nanoseconds)


def save_groud_truth(bag, kitti, pose_frame_id, topic, new_topic):
    print("Exporting Pose")
    x0, y0, z0, scale = 0, 0, 0, 0
    init = False
    if new_topic:
        pose_topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic,
            type='geometry_msgs/msg/PoseStamped',
            serialization_format='cdr')
        bag.create_topic(pose_topic_info)
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        if not init:
            scale = np.cos(oxts.packet.lat * pi / 180.0)
            [x0, y0, z0] = se3_translation(oxts.packet.lat, oxts.packet.lon, oxts.packet.alt, scale)
            init = True
        t = float(timestamp.strftime("%s.%f"))
        timer = Time(seconds=int(t), nanoseconds=int((t - int(t)) * 10 ** 9))
        head = Header(stamp=timer.to_msg(), frame_id=pose_frame_id)
        q = tf_transformations.quaternion_from_euler(oxts.packet.roll, oxts.packet.pitch, oxts.packet.yaw)
        quaternion = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        [xx, yy, zz] = se3_translation(oxts.packet.lat, oxts.packet.lon, oxts.packet.alt, scale)
        p = Point(x=xx - x0, y=yy - y0, z=zz - z0)
        pose = Pose(position=p, orientation=quaternion)
        ps_msg = PoseStamped(header=head, pose=pose)

        bag.write(topic, serialize_message(ps_msg), timer.nanoseconds)


def load_tracklets_for_frames(bag, stamps, xml_path, topic, new_topic):
    if new_topic:
        clock_topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic,
            type='autoware_auto_perception_msgs/msg/DetectedObjects',
            serialization_format='cdr')
        bag.create_topic(clock_topic_info)

    n_frames = len(stamps)
    tracklets = parseXML(xml_path)

    frame_tracklets = {}
    for i in range(n_frames):
        frame_tracklets[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):

        h, w, l = tracklet.size

        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])

        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
                continue

            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T

            min_x = sys.float_info.max
            max_x = -sys.float_info.max
            for element in cornerPosInVelo[0]:
                if element < min_x:
                    min_x = element
                if element > max_x:
                    max_x = element

            min_y = sys.float_info.max
            max_y = -sys.float_info.max
            for element in cornerPosInVelo[1]:
                if element < min_y:
                    min_y = element
                if element > max_y:
                    max_y = element

            x_distance = max_x - min_x
            y_distance = max_y - min_y

            if x_distance >= y_distance:
                bbox_x = l
                bbox_y = w
            else:
                bbox_x = w
                bbox_y = l

            ground_truth_obj = {
                "type": tracklet.objectType,
                "pose_x": translation[0],
                "pose_y": translation[1],
                "pose_z": translation[2] + h/2,
                "bbox_x": l,
                "bbox_y": w,
                "bbox_z": h,
                "rotation_y": rotation[2]
            }

            frame_tracklets[absoluteFrameNumber].append(ground_truth_obj)

    for idx, stamp in enumerate(stamps):

        gt_objects_msg = DetectedObjects()

        t = float(stamp.strftime("%s.%f"))
        timer = Time(seconds=int(t), nanoseconds=int((t - int(t)) * 10 ** 9))
        gt_objects_msg.header.stamp = timer.to_msg()
        gt_objects_msg.header.frame_id = "velodyne_top"

        for gt_object in frame_tracklets[idx]:

            gt_object_msg = DetectedObject()
            gt_object_msg.existence_probability = 1.0
            gt_object_msg.shape.type = Shape.BOUNDING_BOX

            object_classification = ObjectClassification()
            if gt_object['type'] == 'Car':
                object_classification.label = ObjectClassification.CAR
            elif gt_object['type'] == 'Pedestrian':
                object_classification.label = ObjectClassification.PEDESTRIAN
            elif gt_object['type'] == 'Cyclist':
                object_classification.label = ObjectClassification.BICYCLE
            else:
                continue
            gt_object_msg.classification.append(object_classification)

            gt_object_msg.shape.dimensions.x = float(gt_object['bbox_x'])
            gt_object_msg.shape.dimensions.y = float(gt_object['bbox_y'])
            gt_object_msg.shape.dimensions.z = float(gt_object['bbox_z'])

            gt_object_msg.kinematics.pose_with_covariance.pose.position.x = float(gt_object['pose_x'])
            gt_object_msg.kinematics.pose_with_covariance.pose.position.y = float(gt_object['pose_y'])
            gt_object_msg.kinematics.pose_with_covariance.pose.position.z = float(gt_object['pose_z'])

            yaw = gt_object['rotation_y']

            q = tf_transformations.quaternion_from_euler(0, 0, float(gt_object['rotation_y']))

            gt_object_msg.kinematics.pose_with_covariance.pose.orientation.x = q[0]
            gt_object_msg.kinematics.pose_with_covariance.pose.orientation.y = q[1]
            gt_object_msg.kinematics.pose_with_covariance.pose.orientation.z = q[2]
            gt_object_msg.kinematics.pose_with_covariance.pose.orientation.w = q[3]

            gt_objects_msg.objects.append(gt_object_msg)

        bag.write(topic, serialize_message(gt_objects_msg), timer.nanoseconds)


def save_clock(bag, kitti, topic, new_topic):
    print("creating clock data")

    if new_topic:
        clock_topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic,
            type='rosgraph_msgs/msg/Clock',
            serialization_format='cdr')
        bag.create_topic(clock_topic_info)
    start, end = kitti.timestamps[0], kitti.timestamps[-1]
    t = float(start.strftime("%s.%f"))
    nt = float(end.strftime("%s.%f"))
    ts = np.arange(t, nt + 0.2, 0.05)

    np.set_printoptions(suppress=False,
                        formatter={'float_kind': '{:f}'.format})

    for cur_t in ts:
        timer = Time(seconds=int(cur_t), nanoseconds=int((cur_t - int(cur_t)) * 10 ** 9))
        clock_msg = Clock(
            clock=timer.to_msg(),
        )

        bag.write(topic, serialize_message(clock_msg), timer.nanoseconds)


def save_dynamic_tf(bag, kitti, topic, new_topic):
    print("Exporting time dependent transformations")

    if new_topic:
        pose_topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic,
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr')
        bag.create_topic(pose_topic_info)
 
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        tf_oxts_msg = TFMessage()
        tf_oxts_transform = TransformStamped()
        t = float(timestamp.strftime("%s.%f"))
        timer = Time(seconds=int(t), nanoseconds=int((t - int(t)) * 10 ** 9))
        tf_oxts_transform.header.stamp = timer.to_msg()
        tf_oxts_transform.header.frame_id = 'world'
        tf_oxts_transform.child_frame_id = 'base_link'

        transform = (oxts.T_w_imu)
        t = transform[0:3, 3]
        q = tf_transformations.quaternion_from_matrix(transform)
        oxts_tf = Transform()

        oxts_tf.translation.x = t[0]
        oxts_tf.translation.y = t[1]
        oxts_tf.translation.z = t[2]

        oxts_tf.rotation.x = q[0]
        oxts_tf.rotation.y = q[1]
        oxts_tf.rotation.z = q[2]
        oxts_tf.rotation.w = q[3]

        tf_oxts_transform.transform = oxts_tf
        tf_oxts_msg.transforms.append(tf_oxts_transform)

        bag.write(topic, serialize_message(tf_oxts_msg), timer.nanoseconds)


def get_static_transform(from_frame_id, to_frame_id, transform):
    t = transform[0:3, 3]
    q = tf_transformations.quaternion_from_matrix(transform)
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = from_frame_id
    tf_msg.child_frame_id = to_frame_id
    tf_msg.transform.translation.x = float(t[0])
    tf_msg.transform.translation.y = float(t[1])
    tf_msg.transform.translation.z = float(t[2])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def save_static_transforms(bag, transforms, timestamps, topic, new_topic):
    print("Exporting static transformations")
    if new_topic:
        qos_profile = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        pose_topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic,
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr',
            offered_qos_profiles='- history: 1\n  depth: 1\n  reliability: 1\n  durability: 1\n  deadline:\n    sec: 9223372036\n    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n  avoid_ros_namespace_conventions: false'
        )
        bag.create_topic(pose_topic_info)

    tfm = TFMessage()
    for transform in transforms:
        t = get_static_transform(from_frame_id=transform[0], to_frame_id=transform[1], transform=transform[2])
        tfm.transforms.append(t)

    for timestamp in timestamps:
        t = float(timestamp.strftime("%s.%f"))
        timer = Time(seconds=int(t), nanoseconds=int((t - int(t)) * 10 ** 9))
        for i in range(len(tfm.transforms)):
            tfm.transforms[i].header.stamp = timer.to_msg()
        bag.write(topic, serialize_message(tfm), timer.nanoseconds)
        return


# def save_camera_data(bag, kitti, pose_frame_id, topic, new_topic):
#
#     if new_topic:
#         pose_topic_info = rosbag2_py._storage.TopicMetadata(
#             name=topic,
#             type='sensor_msgs/msg/CameraInfo',
#             serialization_format='cdr')
#         bag.create_topic(pose_topic_info)
#
#
#     camera_pad = '{0:02d}'.format(camera)
#     image_dir = os.path.join(kitti.data_path, 'image_{}'.format(camera_pad))
#     image_path = os.path.join(image_dir, 'data')
#     image_filenames = sorted(os.listdir(image_path))
#     with open(os.path.join(image_dir, 'timestamps.txt')) as f:
#         image_datetimes = map(lambda x: datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f'), f.readlines())
#
#     t = float(timestamp.strftime("%s.%f"))
#     timer = Time(seconds=int(t), nanoseconds=int((t - int(t)) * 10 ** 9))
#
#     calib = CameraInfo()
#     calib.header.frame_id = camera_frame_id
#     calib.width, calib.height = tuple(util['S_rect_{}'.format(camera_pad)].tolist())
#     calib.distortion_model = 'plumb_bob'
#     calib.K = util['K_{}'.format(camera_pad)]
#     calib.R = util['R_rect_{}'.format(camera_pad)]
#     calib.D = util['D_{}'.format(camera_pad)]
#     calib.P = util['P_rect_{}'.format(camera_pad)]
#
#     calib.header.stamp = image_message.header.stamp
#     bag.write(topic + '/camera_info', calib, t=calib.header.stamp)
#     bag.write(topic, serialize_message(calib), timer.nanoseconds)




def inv(transform):
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1 * R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv


kitti = pykitti.raw(basedir, date, drive[0], frames=None)

writer = rosbag2_py.SequentialWriter()

storage_options = rosbag2_py._storage.StorageOptions(
    uri="/home/kaan/KITTI/kitti_0093/2011_09_26/bag_file",
    storage_id='sqlite3')

converter_options = rosbag2_py._storage.ConverterOptions('', '')
writer.open(storage_options, converter_options)

clock_topic = "/clock"
save_clock(writer, kitti, clock_topic, True)

imu_frame_id = 'imu_link'
imu_topic = '/kitti/oxts/imu'
gps_fix_topic = '/kitti/oxts/gps/fix'
gps_vel_topic = '/kitti/oxts/gps/vel'
velo_frame_id = 'velodyne_top'
velo_topic = '/kitti/velo'

T_base_link_to_imu = np.eye(4, 4)
T_base_link_to_imu[0:3, 3] = [-2.71 / 2.0 - 0.05, 0.32, 0.93]

# CAMERAS
cameras = [
    (0, 'camera_gray_left', '/kitti/camera_gray_left'),
    (1, 'camera_gray_right', '/kitti/camera_gray_right'),
    (2, 'camera_color_left', '/kitti/camera_color_left'),
    (3, 'camera_color_right', '/kitti/camera_color_right')
]
# tf_static
transforms = [
    ('base_link', imu_frame_id, T_base_link_to_imu),
    (imu_frame_id, velo_frame_id, inv(kitti.calib.T_velo_imu)),
    (imu_frame_id, cameras[0][1], inv(kitti.calib.T_cam0_imu)),
    (imu_frame_id, cameras[1][1], inv(kitti.calib.T_cam1_imu)),
    (imu_frame_id, cameras[2][1], inv(kitti.calib.T_cam2_imu)),
    (imu_frame_id, cameras[3][1], inv(kitti.calib.T_cam3_imu))
]

util = pykitti.utils.read_calib_file(os.path.join(kitti.calib_path, 'calib_cam_to_cam.txt'))

save_static_transforms(writer, transforms, kitti.timestamps, '/tf_static', True)

save_dynamic_tf(writer, kitti, '/tf', True)

xml_path = '/home/kaan/KITTI/kitti_0093/2011_09_26/tracklet_labels.xml'
load_tracklets_for_frames(writer, kitti.timestamps, xml_path, 'gt_objects', True)

velo_topic = '/sensing/lidar/top/rectified/pointcloud'
velo_frame_id = 'velodyne_top'
save_pcl(writer, kitti, velo_frame_id, velo_topic, True)

# save_camera_data(writer, kitti.timestamps, util,  camera[0], camera_frame_id=camera[1], topic=camera[2], initial_time=None)

#!/usr/bin/env python3

import argparse
import functools

import numpy as np
import poseviz
import tensorflow as tf

import cameralib
import data.datasets3d
import matlabfile
import options
import paths
import util
import video_io
from options import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--out-video-dir', type=str)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--inner-batch-size', type=int, default=128)
    parser.add_argument('--viz', action=options.BoolAction)
    options.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    model = tf.saved_model.load(FLAGS.model_path)
    ji3d = get_joint_info(model, skeleton='mpi_inf_3dhp_17')

    predict_fn = functools.partial(
        model.detect_poses_batched, internal_batch_size=FLAGS.inner_batch_size,
        num_aug=FLAGS.num_aug, detector_threshold=0.2, detector_nms_iou_threshold=0.7,
        detector_flip_aug=True, antialias_factor=2, suppress_implausible_poses=False,
        skeleton='mpi_inf_3dhp_17')

    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, write_video=bool(FLAGS.out_video_dir),
        world_up=(0, -1, 0), downscale=4, queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None

    image_relpaths_all = []
    poses_all = []
    intrinsic_matrices = util.load_json(f'{paths.DATA_ROOT}/mupots/camera_intrinsics.json')

    for i_seq in range(1, 21):
        if FLAGS.viz:
            viz.new_sequence()
            if FLAGS.out_video_dir:
                viz.start_new_video(f'{FLAGS.out_video_dir}/TS{i_seq}.mp4', fps=25)

        annotations = matlabfile.load(
            f'{paths.DATA_ROOT}/mupots/TS{i_seq}/annot.mat')['annotations']
        camera = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrices[f'TS{i_seq}'], world_up=(0, -1, 0))
        frame_relpaths = [f'mupots/TS{i_seq}/img_{i:06d}.jpg' for i in range(annotations.shape[0])]
        frame_paths = [f'{paths.DATA_ROOT}/{p}' for p in frame_relpaths]
        frame_gpu, frames_cpu = video_io.image_files_as_tf_dataset(
            frame_paths, batch_size=FLAGS.batch_size, prefetch_gpu=2, tee_cpu=True)
        poses_per_frames = predict_sequence(
            predict_fn, frame_gpu, frames_cpu, len(frame_paths), camera, viz)

        for poses_of_frame, frame_relpath in zip(poses_per_frames, frame_relpaths):
            image_relpaths_all.extend([frame_relpath] * len(poses_of_frame))
            poses_all.extend(poses_of_frame)

    np.savez(
        FLAGS.output_path, image_path=np.stack(image_relpaths_all, axis=0),
        coords3d_pred_world=np.stack(poses_all, axis=0))

    if viz is not None:
        viz.close()


def predict_sequence(predict_fn, frame_gpu, frames_cpu, n_frames, camera, viz):
    predict_fn = functools.partial(
        predict_fn, intrinsic_matrix=camera.intrinsic_matrix[np.newaxis],
        distortion_coeffs=camera.get_distortion_coeffs()[np.newaxis])
    progbar = util.progressbar(total=n_frames)
    poses_per_frame = []

    for frames_b_gpu, frames_b_cpu in zip(frame_gpu, frames_cpu):
        pred = predict_fn(frames_b_gpu)
        poses_per_frame.extend(pred['poses3d'].numpy())
        progbar.update(frames_b_gpu.shape[0])

        if viz is not None:
            for frame, boxes, poses3d in zip(
                    frames_b_cpu, pred['boxes'].numpy(), pred['poses3d'].numpy()):
                viz.update(frame, boxes, poses3d, camera)

    return poses_per_frame


def get_joint_info(model, skeleton):
    joint_names = [b.decode('utf8') for b in model.per_skeleton_joint_names[skeleton].numpy()]
    edges = model.per_skeleton_joint_edges[skeleton].numpy()
    return data.datasets3d.JointInfo(joint_names, edges)


if __name__ == '__main__':
    main()

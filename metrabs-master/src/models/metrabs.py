import einops
import keras
import keras.layers
import keras.metrics
import numpy as np
import tensorflow as tf
from attrdict import AttrDict

import models.eval_metrics
import models.model_trainer
import models.util
import tfu
import tfu3d
from options import FLAGS


class Metrabs(keras.Model):
    def __init__(self, backbone, joint_info):
        super().__init__()
        self.backbone = backbone    # 3개중 선택해 만들은 backbone 가져오고
        self.joint_names = tf.Variable(np.array(joint_info.names), trainable=False)
        self.joint_edges = tf.Variable(np.array(joint_info.stick_figure_edges), trainable=False)
        self.joint_info = joint_info    # joints 관련 정보들 정리하고,
        n_raw_points = 32 if FLAGS.transform_coords else joint_info.n_joints
        self.heatmap_heads = MetrabsHeads(n_points=n_raw_points)    # ?
        if FLAGS.transform_coords:
            self.recombination_weights = tf.constant(np.load('32_to_122'))

    #1 MeTRAbsTrainer에서 init하면서 model만들자마자 call할 때 기작
    """
        inp->   (
                    inp = keras.Input(shape=(None, None, 3), dtype=tfu.get_dtype()),
                    intr = keras.Input(shape=(3, 3), dtype=tf.float32)
                )
        일단 저 둘이 tuple로 묶여서 inp로 들어옴. 어딘가에서 Net의 input으로 쓰일 애들임.
        
        features를 뽑는다.
    """
    def call(self, inp, training=None):
        """
            inp: MeTRAbsTrainer보니깐, 별 의미없는 input인듯
        """
        image, intrinsics = inp
        features = self.backbone(image, training=training)      # features: models.Effnetv2결과인 outputs return!
        coords2d, coords3d = self.heatmap_heads(features, training=training)    # features: backbone result / training: False
                                                                                #  (2d 예측좌표,   3d-logits->2d좌표,3d좌표) 를 반환.
        """
        22.05.19
        main - train - MODEL 에서, scope()내에서 trainer_class 생성할 때 생성자에서 model에 첫 input을 넣고 실행하는데, 그 때 기작을 보는중이었음.   
        model - effnetv2  썼다고 가정함.
        
        """
        coords3d_abs = tfu3d.reconstruct_absolute(coords2d, coords3d, intrinsics)
        if FLAGS.transform_coords:
            coords3d_abs = self.latent_points_to_joints(coords3d_abs)

        return coords3d_abs

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float16),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32)])
    def predict_multi(self, image, intrinsic_matrix):
        # This function is needed to avoid having to go through Keras' __call__
        # in the exported SavedModel, which causes all kinds of problems.
        return self.call((image, intrinsic_matrix), training=False)

    def latent_points_to_joints(self, points):
        return tfu3d.linear_combine_points(points, self.recombination_weights)


class MetrabsHeads(keras.layers.Layer):
    """
        backbone에서 나온거(B 1280 1 1) or (B 1280) 바로 먹어서  좌표 예측해주는 module
    """
    def __init__(self, n_points):   # n_raw_points로 초기화.
        super().__init__()
        self.n_points = n_points    # n_points: n_raw_points = 32 if FLAGS.transform_coords else joint_info.n_joints -> coords바꿀거면 32로, 아니면
        self.n_outs = [self.n_points, FLAGS.depth * self.n_points]
        self.conv_final = keras.layers.Conv2D(filters=sum(self.n_outs), kernel_size=1)  # 1x1 conv fileter 같은데?
        """
        conv_final -> n_out개의 필터가진 1x1_CONV layer
        
        inp(
        """

    def call(self, inp, training=None):
        x = self.conv_final(inp)    # 1x1 conv                                                      #1 conv_final(1x1)을 해준다, inp = backbone 결과중 image
        logits2d, logits3d = tf.split(x, self.n_outs, axis=tfu.channel_axis())                      #  x를 n_outs=[n_points, depth*n_points] 이런길이의 두덩이로 채널 차원에 대해 나눔. / 앞은 2d꺼, 뒤는 3d꺼
        # 급 FLAGS에 뭐 들어오는지 궁금하다. 그렇거 정리된거 없나? -> 여기저기서 parser 사용.
        current_format = 'b h w (d j)' if tfu.get_data_format() == 'NHWC' else 'b (d j) h w'        #  포맷 딱 지정하고,
        logits3d = einops.rearrange(logits3d, f'{current_format} -> b h w d j', j=self.n_points)    #  logits3d -> b h w d j(관절개수 n_points) 로 설정. --> h w d 3D-heatmap이 관절개수만큼 존재. batch 별로.
        coords3d = tfu.soft_argmax(tf.cast(logits3d, tf.float32), axis=[2, 1, 3])                   #2 3d-logits -> 3d-coordinates로 soft-argmax. 좌표 뽑음!
        coords3d_rel_pred = models.util.heatmap_to_metric(coords3d, training)                       #3 2d이미지?랑 coords3d랑 concat함            {heatmap_to_metric? 그리고 왜 3d인데 2d이미지를 만든거지?}
        coords2d = tfu.soft_argmax(tf.cast(logits2d, tf.float32), axis=tfu.image_axes()[::-1])      #4 2d히트맵 좌표뽑기! ,  axis: (W,H)->_DATA_FORMAT.index('H'), _DATA_FORMAT.index('W')를 반대순서로 가져옴
        coords2d_pred = models.util.heatmap_to_image(coords2d, training)                            #  2d좌표를 이미지?로 만들음.
        return coords2d_pred, coords3d_rel_pred                                                     #  (2d 예측좌표->이미지(2d좌표),   3d-logits->2d이미지,3d좌표) 를 반환.
    """
        << 변수 정리 >>
                name                        shape                                  explanation
        inp                         ?
        x                           b h w (2d_out,3d_out)               2d용, 3d용 features concat되어서 나옴. 
        logits3d                    depth*n_points -> b h w d j         
            coords3d                soft-argmax    -> b j               hwd 차원에 대해 soft-argmax했음.
            coords3d_rel_pred       [2dcoords][3dcoords]                결과좌표를 image scale, metric scale로 변환 
        logits2d                    n_points(b h w j)                   
            coords2d                ( b j )                             예측 결과 soft-argmax함
            coords2d_pred                                               2d결과 좌표를 image scale 변환해준
        
    """


class MetrabsTrainer(models.model_trainer.ModelTrainer):
    def __init__(self, metrabs_model, joint_info, joint_info2d=None, global_step=None):
        super().__init__(global_step)
        self.global_step = global_step
        self.joint_info = joint_info
        self.joint_info_2d = joint_info2d
        self.model = metrabs_model  # 만들어진 객체 들어옴
        inp = keras.Input(shape=(None, None, 3), dtype=tfu.get_dtype())
        intr = keras.Input(shape=(3, 3), dtype=tf.float32)
        self.model((inp, intr), training=False)         # 들어온 객체 input, intrinsic 퍼즐 넣어줌 -- inp: image(H W 3) / intr(3 3): intrinsics 로 call() or ?

    # 왜 안쓰지? 대신 fit 쓴건가?
    def forward_train(self, inps, training):
        preds = AttrDict()

        image_both = tf.concat([inps.image, inps.image_2d], axis=0)
        features = self.model.backbone(image_both, training=training)
        coords2d_pred_both, coords3d_rel_pred_both = self.model.heatmap_heads(
            features, training=training)
        batch_sizes = [t.shape.as_list()[0] for t in [inps.image, inps.image_2d]]
        preds.coords2d_pred, preds.coords2d_pred_2d = tf.split(
            coords2d_pred_both, batch_sizes, axis=0)
        preds.coords3d_rel_pred, preds.coords3d_rel_pred_2d = tf.split(
            coords3d_rel_pred_both, batch_sizes, axis=0)

        if FLAGS.transform_coords:
            l2j = self.model.latent_points_to_joints
            preds.coords2d_pred_2d = l2j(preds.coords2d_pred_2d)
            preds.coords3d_rel_pred_2d = l2j(preds.coords3d_rel_pred_2d)
            preds.coords2d_pred_latent = preds.coords2d_pred
            preds.coords2d_pred = l2j(preds.coords2d_pred_latent)
            preds.coords3d_rel_pred_latent = preds.coords3d_rel_pred
            preds.coords3d_rel_pred = l2j(preds.coords3d_rel_pred_latent)
            preds.coords3d_pred_abs = l2j(tfu3d.reconstruct_absolute(
                preds.coords2d_pred_latent, preds.coords3d_rel_pred_latent, inps.intrinsics))
        else:
            preds.coords3d_pred_abs = tfu3d.reconstruct_absolute(
                preds.coords2d_pred, preds.coords3d_rel_pred, inps.intrinsics)

        joint_ids_3d = [
            [self.joint_info.ids[n2] for n2 in self.joint_info.names if n2.startswith(n1)]
            for n1 in self.joint_info_2d.names]

        def get_2dlike_joints(coords):
            return tf.stack(
                [tf.reduce_mean(tf.gather(coords, ids, axis=1)[..., :2], axis=1)
                 for ids in joint_ids_3d], axis=1)

        # numbers mean: 3d head, 2d dataset joints, 2d batch
        preds.coords32d_pred_2d = get_2dlike_joints(preds.coords3d_rel_pred_2d)
        preds.coords22d_pred_2d = get_2dlike_joints(preds.coords2d_pred_2d)
        return preds

    def compute_losses(self, inps, preds):
        losses = AttrDict()

        if FLAGS.scale_agnostic_loss:
            mean_true, scale_true = tfu.mean_stdev_masked(
                inps.coords3d_true, inps.joint_validity_mask, items_axis=1, dimensions_axis=2)
            mean_pred, scale_pred = tfu.mean_stdev_masked(
                preds.coords3d_rel_pred, inps.joint_validity_mask, items_axis=1, dimensions_axis=2)
            coords3d_pred_rootrel = tf.math.divide_no_nan(
                preds.coords3d_rel_pred - mean_pred, scale_pred) * scale_true
            coords3d_true_rootrel = inps.coords3d_true - mean_true
        else:
            coords3d_true_rootrel = tfu3d.center_relative_pose(
                inps.coords3d_true, inps.joint_validity_mask, FLAGS.mean_relative)
            coords3d_pred_rootrel = tfu3d.center_relative_pose(
                preds.coords3d_rel_pred, inps.joint_validity_mask, FLAGS.mean_relative)

        rootrel_absdiff = tf.abs((coords3d_true_rootrel - coords3d_pred_rootrel) / 1000)
        losses.loss3d = tfu.reduce_mean_masked(rootrel_absdiff, inps.joint_validity_mask)

        if FLAGS.scale_agnostic_loss:
            _, scale_true = tfu.mean_stdev_masked(
                inps.coords3d_true, inps.joint_validity_mask, items_axis=1, dimensions_axis=2,
                fixed_ref=tf.zeros_like(inps.coords3d_true))
            _, scale_pred = tfu.mean_stdev_masked(
                preds.coords3d_pred_abs, inps.joint_validity_mask, items_axis=1, dimensions_axis=2,
                fixed_ref=tf.zeros_like(inps.coords3d_true))
            preds.coords3d_pred_abs = tf.math.divide_no_nan(
                preds.coords3d_pred_abs, scale_pred) * scale_true

        if self.global_step > 5000:
            absdiff = tf.abs((inps.coords3d_true - preds.coords3d_pred_abs) / 1000)
            losses.loss3d_abs = tfu.reduce_mean_masked(absdiff, inps.joint_validity_mask)
        else:
            losses.loss3d_abs = tf.constant(0, tf.float32)

        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
        losses.loss23d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true - preds.coords2d_pred) * scale_2d),
            inps.joint_validity_mask)

        preds.coords32d_pred_2d = models.util.align_2d_skeletons(
            preds.coords32d_pred_2d, inps.coords2d_true_2d, inps.joint_validity_mask_2d)
        losses.loss32d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true_2d - preds.coords32d_pred_2d) * scale_2d),
            inps.joint_validity_mask_2d)
        losses.loss22d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true_2d - preds.coords22d_pred_2d) * scale_2d),
            inps.joint_validity_mask_2d)

        losses3d = [losses.loss3d, losses.loss23d, FLAGS.absloss_factor * losses.loss3d_abs]
        losses2d = [losses.loss22d, losses.loss32d]
        losses.loss = tf.add_n(losses3d) + FLAGS.loss2d_factor * tf.add_n(losses2d)
        return losses

    def compute_metrics(self, inps, preds):
        return models.eval_metrics.compute_pose3d_metrics(inps, preds)

    def forward_test(self, inps):
        preds = AttrDict()
        features = self.model.backbone(inps.image, training=False)
        preds.coords2d_pred, preds.coords3d_rel_pred = self.model.heatmap_heads(
            features, training=False)
        preds.coords3d_pred_abs = tfu3d.reconstruct_absolute(
            preds.coords2d_pred, preds.coords3d_rel_pred, inps.intrinsics)

        if FLAGS.transform_coords:
            l2j = self.model.latent_points_to_joints
            preds.coords2d_pred = l2j(preds.coords2d_pred)
            preds.coords3d_rel_pred = l2j(preds.coords3d_rel_pred)
            preds.coords3d_pred_abs = l2j(preds.coords3d_pred_abs)

        return preds

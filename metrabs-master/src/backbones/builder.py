import functools

import numpy as np
import tensorflow as tf
from keras.layers import Lambda
from keras.models import Sequential

import backbones.efficientnet.effnetv2_model as effnetv2_model
import backbones.efficientnet.effnetv2_utils as effnetv2_utils
import backbones.mobilenet_v3
import backbones.resnet
import keras.layers
import tfu
from layers.custom_batchnorms import GhostBatchNormalization
from options import FLAGS


def build_backbone():
    """
    기능: efficientnetv2, resnet, mobilenet 중 하나 선택해서 만들어 반환. FLAGS의 인자 참고해서 만듦.
    reuturn: [input rgb 정규화 layer -- backbone] 으로 구성된 Sequential 클래스를 반환함.
    """
    if FLAGS.backbone.startswith('efficientnetv2'):     # efficientnetv2 로 backbone argument가 시작하면,
        bn = effnetv2_utils.BatchNormalization          # efficientnetv2 의 batchnormalization 클래스를 사용한다.(bn이 클래스임.)
    else:
        bn = keras.layers.BatchNormalization            # 아니면 keras의 기본 BN 쓰는 듯 하다.

    if FLAGS.ghost_bn:  # BN technique인듯. 뭐지근데?
        split = list(map(int, FLAGS.ghost_bn.split(',')))
        bn = functools.partial(GhostBatchNormalization, split=split)

    if FLAGS.backbone.startswith('efficientnetv2'):                                 # backbone에 입력한 argument가 effnetv2면,
        effnetv2_utils.set_batchnorm(bn)                                                # effnetv2_utils의 _BatchNorm 전역변수를 bn(클래스)로 바꿈.
        backbone = effnetv2_model.get_model(FLAGS.backbone, include_top=False)          # backbone: 사전학습된 keras로 만들어진 EffNet 모델(클래스 객체) 저장!
        preproc_fn = tf_preproc
    elif FLAGS.backbone.startswith('resnet'):                                       # resnet이면?
        class MyLayers(keras.layers.VersionAwareLayers):                                # 갑자기 MyLayers를 만든다, BatchNormalization 호환성 때문에 급히 만든듯.
            def __getattr__(self, name):
                if name == 'BatchNormalization':
                    return bn
                return super().__getattr__(name)

        classname = f'ResNet{FLAGS.backbone[len("resnet"):]}'.replace('-', '_')
        backbone = getattr(backbones.resnet, classname)(                                # backbone: resnet 저장!
            include_top=False, weights='imagenet',
            input_shape=(None, None, 3), layers=MyLayers())
        if 'V2' in FLAGS.backbone:
            preproc_fn = tf_preproc
        elif 'V1-5' in FLAGS.backbone or 'V1_5' in FLAGS.backbone:
            preproc_fn = torch_preproc                                                  # preproc_fn = torch_preproc: tensor에 대해 RGB 뭔가 전처리하는 함수! 다른애들도 same
        else:
            preproc_fn = caffe_preproc                                                  # RGB 정규화
    elif FLAGS.backbone.startswith('mobilenet'):                                    # mobilenet이면?
        class MyLayers(backbones.mobilenet_v3.VersionAwareLayers):
            def __getattr__(self, name):
                if name == 'BatchNormalization':
                    return bn
                return super().__getattr__(name)

        arch = FLAGS.backbone
        arch = arch[:-4] if arch.endswith('mini') else arch
        classname = f'MobileNet{arch[len("mobilenet"):]}'
        backbone = getattr(backbones.mobilenet_v3, classname)(                              # backbone <- mobilenet 저장!
            include_top=False, weights='imagenet', minimalistic=FLAGS.backbone.endswith('mini'),
            input_shape=(FLAGS.proc_side, FLAGS.proc_side, 3), layers=MyLayers(),
            centered_stride=FLAGS.centered_stride, pooling=None)
        preproc_fn = mobilenet_preproc
    else:                                                                           # 셋다 아니면? 예외처리
        raise Exception('Choose backbone plz~~~ or I am going to break your backbone~~~')

    return Sequential([Lambda(preproc_fn, output_shape=lambda x: x), backbone])     # [preproc_fn layer(rgb 정규화) -> backbone(3가지중 하나)] 의 Sequential 네트워크를 반환!
                                                                                    # input -> <Tensor>, output -> <backbone 통과 결과>
    # Keras.model.Lambda -> 임의의 함수를 layer로 만들어 쓸 수 있게한다. (tf.keras.layers.Lambda() 의 기능임)



"""
얘들 뭐냐?
"""

# torch 전처리? 뭔가 RGB에 대해 정규화 해준다.
def torch_preproc(x):
    """
    input -> Tensor
    process -> Normalizing Tensor
    output -> Tensor
    """
    mean_rgb = tf.convert_to_tensor(np.array([0.485, 0.456, 0.406]), tfu.get_dtype())   # tf.convert_to_tensor():<np.array> 를 TF의 2개의 tensor중 하나인 grad저장 안하는 tensor로 바꿔주는 애
    stdev_rgb = tf.convert_to_tensor(np.array([0.229, 0.224, 0.225]), tfu.get_dtype())
    normalized = (x - mean_rgb) / stdev_rgb
    return normalized


def caffe_preproc(x):
    mean_rgb = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68]), tfu.get_dtype())
    return tf.cast(255, tfu.get_dtype()) * x - mean_rgb


def tf_preproc(x):
    x = tf.cast(2, tfu.get_dtype()) * x - tf.cast(1, tfu.get_dtype())
    return x


def mobilenet_preproc(x):
    return tf.cast(255, tfu.get_dtype()) * x

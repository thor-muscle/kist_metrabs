#!/usr/bin/env python3
import contextlib
import os
import re
import sys

import attrdict
import keras
import keras.callbacks
import keras.models
import keras.optimizers
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import backbones.builder
import callbacks
import data.data_loading
import data.datasets2d
from data.datasets2d import get_dataset
import init
import models.metrabs
import models.util
import parallel_preproc
import tfu
import util
from options import FLAGS, logger
from tfu import TEST, TRAIN, VALID


def train():
    strategy = tf.distribute.MirroredStrategy() if FLAGS.multi_gpu else dummy_strategy()    # multi-gpu
    n_repl = strategy.num_replicas_in_sync  # n_repl: GPU개수  ->  replicas= GPU / num_replicas_in_sync: Returns number of replicas over which gradients are aggregated.


    ######
    # TRAINING DATA
    #######

    dataset3d = data.datasets3d.get_dataset(FLAGS.dataset)  # 내가 지정한 dataset을 가져오나봄. : dataset
    joint_info3d = dataset3d.joint_info
    examples3d = get_examples(dataset3d, TRAIN, FLAGS)      # dataset3d 의 examples를 train모드로 가져옴. : dataset -> train mode

    dataset2d = data.datasets2d.get_dataset(FLAGS.dataset2d)    # ?
    joint_info2d = dataset2d.joint_info
    examples2d = [*dataset2d.examples[TRAIN], *dataset2d.examples[VALID]]   # : dataset2d -> [train data, valid data] <list>

    # only for 3d data
    # dataset 여러개 선택한 경우 -> example들을 이어붙이고, section을 구분해서 알아놓는다.
    if 'many' in FLAGS.dataset:
        if 'aist' in FLAGS.dataset:     # gta5 + 춤추는 3D 모션 데이터셋
            dataset_section_names = 'h36m muco-3dhp surreal panoptic aist_ sailvos'.split()
            roundrobin_sizes = [8, 8, 8, 8, 8, 8]       # 시분할 시스템 방식 : roundrobin
            roundrobin_sizes = [x * 2 for x in roundrobin_sizes]
        else:           # GTA5 sail-vos dataset
            dataset_section_names = 'h36m muco-3dhp panoptic surreal sailvos'.split()
            roundrobin_sizes = [9, 9, 9, 9, 9]
        example_sections = build_dataset_sections(examples3d, dataset_section_names)    # examples3d are ordered by "dataset_section_names" : dataset -> train mode -> ordered by data_name<list>
    else:
        example_sections = [examples3d]                                                 # using single? dataset. 자료형은 'many'처럼 list로 맞춰줌.
        roundrobin_sizes = [FLAGS.batch_size]

    n_completed_steps = get_n_completed_steps(FLAGS.checkpoint_dir, FLAGS.load_path)    # 이전에 학습된 steps?

    rng = np.random.RandomState(FLAGS.seed)
    data2d = build_dataflow(
        examples2d, data.data_loading.load_and_transform2d, (joint_info2d, TRAIN),
        TRAIN, batch_size=FLAGS.batch_size_2d * n_repl, n_workers=FLAGS.workers,
        rng=util.new_rng(rng), n_completed_steps=n_completed_steps,
        n_total_steps=FLAGS.training_steps)
    # dataset2d -> [t,v]<list> -> 병렬처리, 배치설정된<tf.data.Dataset>
    data3d = build_dataflow(
        example_sections, data.data_loading.load_and_transform3d, (joint_info3d, TRAIN),
        tfu.TRAIN, batch_size=sum(roundrobin_sizes)//2 * n_repl,
        n_workers=FLAGS.workers,
        rng=util.new_rng(rng), n_completed_steps=n_completed_steps,
        n_total_steps=FLAGS.training_steps, roundrobin_sizes=roundrobin_sizes)
    # dataset3d -> train mode -> ordered by data_name<list> -> 병렬처리, 배치설정된<tf.data.Dataset>
    """
    dataflow를 만들음 == 병렬처리, 배치설정된 Dataset 객체
    """

    data_train = tf.data.Dataset.zip((data3d, data2d))  # 데이터 묶기 -> data_train은 (data3d,data2d) 쌍으로 이루어진 Dataset 객체임. [O]
                                                        #           -> tf.data.Dataset 내부에서 zip은 dict로 저장하도록 했어야한다. 왜냐하면 원소들이 dict형대로 저장되어야 해서. [X]
    """
    tf.data.Dataset 객체를 built-in zip()처럼 생각해버림. 실상은, data3d 라는 tf.data.Dataset 객체와 data2d 라는 같은 Namespace 객체를 zip()해준거다.
    data_train -> [[ (data3d[0], data2d[0]), (data3d[1],data2d[1]), ... ]]
    """
    # dataset2d -> [t,v]<list> -> 병렬처리, 배치설정된<tf.data.Dataset>
    # dataset3d -> train mode -> ordered by data_name<list> -> 병렬처리, 배치설정된<tf.data.Dataset>
    # data_train: Dataset 객체 <tf.data.Dataset> [(3d,2d), (3d,2d), ... ] 꼴.
    data_train = data_train.map(lambda batch3d, batch2d: {**batch3d, **batch2d})    # data_train꼴 변경 -> data_train의 모든 원소들에 대해 저 연산?을 수행. 뭐하는거지?
                                                                                    # data_train의 원소들이 2 element tuple형태로 존재하니깐, 저렇게 바꾼건가?
                                                                                    # data_train의 원소가 dict형태일 경우에만 저게 가능하다.

                                                                                    # 뭘 하고싶었던걸까?
    """
    map(map_func, num_parallel_calls=None, deterministic=None, name=None) 으로 정의됨
    Maps map_func across the elements of this dataset.
    This transformation applies map_func to each element of this dataset, and returns a new dataset containing the transformed elements, 
    in the same order as they appeared in the input. map_func can be used to change both the values and the structure of a dataset's elements.
    Supported structure constructs are documented here.
    For example, map can be used for adding 1 to each element, or projecting a subset of element components.
    
    그만하고, 중요한 시퀀스만 파악하고, 고쳐야할부분 아닌부분 나눠서 생각해보자.
    -> data_train의 형태가 dict로 변했다?
    """
    # 쌍으로 연결된 원소들을 dict 모양으로 풀어서 연결시킴.
    # 뭔가 아까 batch먹여서 그거 고려되어서 map되는것 같음.
    # 추측!!! -> data_train: Dataset [{data3d + data2d -- (batch개수로 됐을듯)}[0], {data3d + data2d}[1], ...] 각각의 원소를 하나의 dict로 이어붙였음<list>

    if not FLAGS.multi_gpu:
        data_train = data_train.apply(tf.data.experimental.prefetch_to_device('GPU:0', 2))  # 분산학습 시 데이터셋 모양 변경

    # 마지막으로 train data에 option 뭐 해준다.
    opt = tf.data.Options()
    opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA   # 하나의 옵션 추가 : AutoShardPolicy - DATA
    """
    Shard(데이터 수평분할처리) 의 5개의 데이터 처리방식중 하나인 DATA.
    DATA: Shards by elements produced by the dataset.
    Each worker will process the whole dataset and discard the portion that is not for itself.
    Note that for this mode to correctly partitions the dataset elements, 
    the dataset needs to produce elements in a deterministic order.
    """
    data_train = data_train.with_options(opt)   # data_train : <tf.data.Dataset>[{data3d + data2d}[0], {data3d + data2d}[1], ...]
                                                #               batch 고려, Shard policy 적용, Parallel processing 적용된 dataset임.
                                                #               3d랑 2d 합쳐져있음 batch마다.


    #######
    # VALIDATION DATA
    #######
    examples3d_val = get_examples(dataset3d, VALID, FLAGS)  # dataset3d 가져온걸, valid모드로 FLAGS 참고해서 잘 가져온다.

    # val 주기 정하는 듯.
    if FLAGS.validate_period:   # validate_period 인자로 넣은 경우에만 val 진행된다. 왜냐하면 229줄 fit에서 data_val만 추론 데이터로 사용해서.
        data_val = build_dataflow(
            examples3d_val, data.data_loading.load_and_transform3d,
            (joint_info3d, VALID), VALID, batch_size=FLAGS.batch_size_test * n_repl,
            n_workers=FLAGS.workers, rng=util.new_rng(rng))                             # data_val: examples3d_val 을 배치, 병렬처리 먹인다.<tf.data.Dataset>
        data_val = data_val.with_options(opt)                                           # data_cal: 옵션 추가<Dataset>
        validation_steps = int(np.ceil(len(examples3d_val) / (FLAGS.batch_size_test * n_repl)))     # validation_steps: val 횟수 계산.
    else:
        data_val = None
        validation_steps = None


    # argument 받은거 바탕으로 train dataset, val dataset 만들었음.
    # train은 2d, 3d data로 진행함.
    # val은 3d data로만 진행함.
    # 둘 다 Dataset 객체. 병렬처리, 배치, 옵션(Shard) 먹음.



    #######
    # MODEL
    #######
    with strategy.scope():
        """
        이 구문의 역할: Context manager for distributed learning, 모든 gpu가 잘 작동위해 이 내부를 scope하는 중이라 생각하샘
        sync strategy의 경우, 여기서 만들어지는 모든 변수는 분산학습되는 모든 GPU에서 동일하게 생김. 이렇게 변수를 strategy에 맞게 관리함.
        """
        global_step = tf.Variable(n_completed_steps, dtype=tf.int32, trainable=False)   # global_step: 이전에 완료된 step 부터 집계?
        backbone = backbones.builder.build_backbone()                                   # FLAGS에서 인자 받아와서 3가지 모델(effnetv2, resnet, mobilenet)중 하나 선택해 pretrained된 backbone 만든다.

        model_class = getattr(models, FLAGS.model_class)                                # model_class: models.py 중에서 Flags에서 말한 class로 설정
        trainer_class = getattr(models, FLAGS.model_class + 'Trainer')                  # trainer_class: 이하동문.
        # getattr(object, name) : object의 name 속성의 값을 가져온다. 여기의 경우는 아래와 같다.
        """
            from models.metrabs import Metrabs, MetrabsTrainer
            from models.metro import Metro, MetroTrainer
            from models.twofive import Model25D, Model25DTrainer
            # from models.twofive_diffble import Model25DDiffble, Model25DDiffbleTrainer   이런거 없음ㅋㅋㅋ
            from models.twofive_tro import Model25DDiffbleTruncRob, Model25DDiffbleTruncRobTrainer
            
            models가 저 models.py 말한거였음.
            models에서 FLAGS.model_class 이름의 속성인 클래스를 model_class로 지정한다.
        """
        bone_lengths = (dataset3d.trainval_bones if FLAGS.train_on == 'trainval' else dataset3d.train_bones)        # bone_lengths: raw_dataset의 att중 하나인 bones(뼈)에 관한 속성을 불러옴. (train, trainval 별로) -- extra_args를 위한 변수
        extra_args = [bone_lengths] if FLAGS.model_class.startswith('Model25D') else []                             # extra_args: twofive.py 에서 가져오는 모델들에게만 필요한 녀석.
        model = model_class(backbone, joint_info3d, *extra_args)                                                # 앞에서 가져온 model 클래스에 앞선 backbone 넣고, joint_info3d넣고, twofive.py에서 가져온 Model25D 같은 경우는 bone_lengths 까지 넣어줌.
        # 바로 위에, *extra_args 를 안받는 model_class 들도 있는데 저렇게 해도 괜찮은거야? [] 들어갈텐데 말이야. -> 확인 해보기
        trainer = trainer_class(model, joint_info3d, joint_info2d, global_step)                                 # 저렇게 4가지를 param으로 받음. 여기서, model <- MeTRAbs(backbone, joint_info3d) 클래스 객체임.
        ################################################!!!!!!!!#####################################바로 위에부터!!
        trainer.compile(optimizer=build_optimizer(global_step, n_repl))
        model.optimizer = trainer.optimizer







    #######
    # CHECKPOINTING
    #######
    ckpt = tf.train.Checkpoint(model=model) # saving, restoring
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        directory=FLAGS.checkpoint_dir,
        max_to_keep=2,
        step_counter=global_step,
        checkpoint_interval=FLAGS.checkpoint_period)
    restore_if_ckpt_available(ckpt, ckpt_manager, global_step, FLAGS.init_path)
    trainer.optimizer.iterations.assign(n_completed_steps)







    #######
    # CALLBACKS
    #######
    cbacks = [
        keras.callbacks.LambdaCallback(
            on_train_begin=lambda logs: trainer._train_counter.assign(n_completed_steps),
            on_train_batch_end=lambda batch, logs: ckpt_manager.save(global_step)),
        callbacks.ProgbarCallback(n_completed_steps, FLAGS.training_steps),
        callbacks.WandbCallback(global_step),
        callbacks.TensorBoardCallback(global_step)
    ]

    if FLAGS.finetune_in_inference_mode:
        switch_step = FLAGS.training_steps - FLAGS.finetune_in_inference_mode
        c = callbacks.SwitchToInferenceModeCallback(global_step, switch_step)
        cbacks.append(c)








    #######
    # FITTING
    #######
    try:
        trainer.fit(
            data_train, steps_per_epoch=1, initial_epoch=n_completed_steps,
            epochs=FLAGS.training_steps, verbose=1 if sys.stdout.isatty() else 0,
            callbacks=cbacks, validation_data=data_val, validation_freq=FLAGS.validate_period,
            validation_steps=validation_steps)
        model.save(
            f'{FLAGS.checkpoint_dir}/model', include_optimizer=False, overwrite=True,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
    except KeyboardInterrupt:
        logger.info('Training interrupted.')
    except tf.errors.ResourceExhaustedError:
        logger.info('Resource Exhausted!')
    finally:
        ckpt_manager.save(global_step, check_interval=False)
        logger.info('Saved checkpoint.')






def build_optimizer(global_step, n_replicas):
    def weight_decay():
        lr_ratio = lr_schedule(global_step) / FLAGS.base_learning_rate
        # Decay the weight decay itself over time the same way as the learning rate is decayed.
        # Division by sqrt(num_training_steps) is taken from the original AdamW paper.
        return FLAGS.weight_decay * lr_ratio / np.sqrt(FLAGS.training_steps)

    def lr():
        return lr_schedule(global_step) / tf.sqrt(tf.cast(n_replicas, tf.float32))

    optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay, learning_rate=lr, epsilon=1e-8)
    # Make sure these exist so checkpoints work properly
    optimizer.iterations
    optimizer.beta_1

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        optimizer, dynamic=False, initial_scale=128)
    return optimizer





# examples을 내가 원하는 section_names을 받아 해당하는 examples의 image_path로 분류해서 list로 반환해주는 function.
# 내가 원하는 section별로 examples를 분류해서 반환하는 애. 정하지 않은 section은 버린다. 없는 section 선택하면 RuntimeError뜸
def build_dataset_sections(examples, section_names):
    """
    input : examples(Dataset), section_names(list)
    output : [section1_examples, section2_examples, section3_examples, ... ] -> ordered by following input section_names
    """
    sections = {name: [] for name in section_names}
    for ex in examples:
        for name in section_names:
            if name in ex.image_path.lower():
                sections[name].append(ex)
                break
        else:       # for가 끊기지 않고 끝까지 수행됐을 때 실행됨, for문 끝까지 실행됐는지 확인위함
            print(f'{name}은 이 example에 없는 section임')
            raise RuntimeError      # 여기선, setcion_names의 모든 이름중 지금 특정 ex.image_path의 이름이 없는경우 에러뜨게 만듦.
    return [sections[name] for name in section_names]





@tf.function
def lr_schedule(global_step):
    training_steps = FLAGS.training_steps
    n_phase1_steps = 0.92 * training_steps
    n_phase2_steps = training_steps - n_phase1_steps
    global_step_float = tf.cast(global_step, tf.float32)
    b = tf.constant(FLAGS.base_learning_rate, tf.float32)

    if global_step_float < n_phase1_steps:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            b, decay_rate=1 / 3, decay_steps=n_phase1_steps, staircase=False)(global_step_float)
    else:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            b * tf.cast(1 / 30, tf.float32), decay_rate=0.3, decay_steps=n_phase2_steps,
            staircase=False)(global_step_float - n_phase1_steps)




# If non-distributed learning stratage
def dummy_strategy():
    @contextlib.contextmanager
    def dummy_scope():
        yield

    return attrdict.AttrDict(scope=dummy_scope, num_replicas_in_sync=1)     # attrdict = can access key as attribute





# ~~~모델을 저장하지.
def export():
    dataset3d = data.datasets3d.get_dataset(FLAGS.dataset)
    ji = dataset3d.joint_info
    del dataset3d
    backbone = backbones.builder.build_backbone()
    model = models.metrabs.Metrabs(backbone, ji)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, None)
    restore_if_ckpt_available(ckpt, ckpt_manager, expect_partial=True)

    if FLAGS.load_path:
        load_path = util.ensure_absolute_path(FLAGS.load_path, FLAGS.checkpoint_dir)
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        load_path = ckpt.model_checkpoint_path

    checkpoint_dir = os.path.dirname(load_path)
    out_path = util.ensure_absolute_path(FLAGS.export_file, checkpoint_dir)
    # 지정된 경로로 model을 저장한다.
    model.save(
        out_path, include_optimizer=False, overwrite=True,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))





def predict():
    dataset3d = data.datasets3d.get_dataset(FLAGS.dataset)
    backbone = backbones.builder.build_backbone()
    model_class = getattr(models, FLAGS.model_class)
    trainer_class = getattr(models, FLAGS.model_class + 'Trainer')
    model_joint_info = data.datasets3d.get_joint_info(FLAGS.model_joints)

    if FLAGS.model_class.startswith('Model25D'):
        bone_dataset = data.datasets3d.get_dataset(FLAGS.bone_length_dataset)
        bone_lengths = (
            bone_dataset.trainval_bones if FLAGS.train_on == 'trainval'
            else bone_dataset.train_bones)
        extra_args = [bone_lengths]
    else:
        extra_args = []
    model = model_class(backbone, model_joint_info, *extra_args)
    trainer = trainer_class(model, model_joint_info)
    trainer.predict_tensor_names = [
        'coords3d_rel_pred', 'coords3d_pred_abs', 'rot_to_world', 'cam_loc', 'image_path']

    if FLAGS.viz:
        trainer.predict_tensor_names += ['image', 'coords3d_true']

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, None)
    restore_if_ckpt_available(ckpt, ckpt_manager, expect_partial=True)

    examples3d_test = get_examples(dataset3d, tfu.TEST, FLAGS)
    data_test = build_dataflow(
        examples3d_test, data.data_loading.load_and_transform3d,
        (dataset3d.joint_info, TEST), TEST, batch_size=FLAGS.batch_size_test,
        n_workers=FLAGS.workers)
    n_predict_steps = int(np.ceil(len(examples3d_test) / FLAGS.batch_size_test))

    r = trainer.predict(
        data_test, verbose=1 if sys.stdout.isatty() else 0, steps=n_predict_steps)
    r = attrdict.AttrDict(r)
    util.ensure_path_exists(FLAGS.pred_path)

    logger.info(f'Saving predictions to {FLAGS.pred_path}')
    try:
        coords3d_pred = r.coords3d_pred_abs
    except AttributeError:
        coords3d_pred = r.coords3d_rel_pred

    coords3d_pred_world = tf.einsum(
        'nCc, njc->njC', r.rot_to_world, coords3d_pred) + tf.expand_dims(r.cam_loc, 1)
    coords3d_pred_world = models.util.select_skeleton(
        coords3d_pred_world, model_joint_info, FLAGS.output_joints).numpy()
    np.savez(FLAGS.pred_path, image_path=r.image_path, coords3d_pred_world=coords3d_pred_world)




# examples<list>를 메모리 병렬처리, 배치 먹임.
def build_dataflow(
        examples, load_fn, extra_args, learning_phase, batch_size, n_workers, rng=None,
        n_completed_steps=0, n_total_steps=None, n_test_epochs=1, roundrobin_sizes=None):
    # 전체 아이템 수 계산
    if learning_phase == tfu.TRAIN:
        n_total_items = int(n_total_steps * batch_size if n_total_steps is not None else None)
    elif learning_phase == tfu.VALID:
        n_total_items = None
    else:
        n_total_items = int(len(examples) * n_test_epochs)

    # 메모리 병렬처리(parallel-preprocessor - PPP)용 Tensorflow Dataset객체로 만들은 듯.
    # <tf.data.Dataset>
    dataset = parallel_preproc.parallel_map_as_tf_dataset(
        load_fn, examples, shuffle_before_each_epoch=(learning_phase == tfu.TRAIN),
        extra_args=extra_args, n_workers=n_workers, rng=rng, max_unconsumed=batch_size * 2,
        n_completed_items=n_completed_steps * batch_size, n_total_items=n_total_items,
        roundrobin_sizes=roundrobin_sizes)

    # "examples" ->"배치고려 후처리된 dataset" return
    return dataset.batch(batch_size, drop_remainder=(learning_phase == tfu.TRAIN))




# 얘가 반환하는 examples3d는 어떤 애지?
# dataset의 examples attribute를 learning_phase에 따라서 조회해 corresponding examples로 반환하는 것
# tfu.Train(0), Valid(1), Test(2)
def get_examples(dataset, learning_phase, flags):
    if learning_phase == tfu.TRAIN:
        str_example_phase = flags.train_on
    elif learning_phase == tfu.VALID:
        str_example_phase = flags.validate_on
    elif learning_phase == tfu.TEST:
        str_example_phase = flags.test_on
    else:
        raise Exception(f'No such learning_phase as {learning_phase}')

    if str_example_phase == 'train':
        examples = dataset.examples[tfu.TRAIN]  # tfu.TRAIN = 0
    elif str_example_phase == 'valid':
        examples = dataset.examples[tfu.VALID]  # 1
    elif str_example_phase == 'test':
        examples = dataset.examples[tfu.TEST]   # 2
    elif str_example_phase == 'trainval':
        examples = [*dataset.examples[tfu.TRAIN], *dataset.examples[tfu.VALID]]     # 두개 이어붙여줌. [ [TRAIN] [VAL] ] 꼴임.
    else:
        raise Exception(f'No such phase as {str_example_phase}')
    return examples




# 뭐 이전에 얼마나 학습했나 저장하는 용도인가?
def get_n_completed_steps(logdir, load_path):
    if load_path is not None:
        return int(re.search('ckpt-(?P<num>\d+)', os.path.basename(load_path))['num'])

    if os.path.exists(f'{logdir}/checkpoint'):
        text = util.read_file(f'{logdir}/checkpoint')
        return int(re.search('model_checkpoint_path: "ckpt-(?P<num>\d+)"', text)['num'])
    else:
        return 0





def restore_if_ckpt_available(
        ckpt, ckpt_manager, global_step_var=None, initial_checkpoint_path=None,
        expect_partial=False):
    resuming_checkpoint_path = FLAGS.load_path
    if resuming_checkpoint_path:
        if resuming_checkpoint_path.endswith('.index'):
            resuming_checkpoint_path = os.path.splitext(resuming_checkpoint_path)[0]
        if not os.path.isabs(resuming_checkpoint_path):
            resuming_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, resuming_checkpoint_path)
    else:
        resuming_checkpoint_path = ckpt_manager.latest_checkpoint

    load_path = resuming_checkpoint_path if resuming_checkpoint_path else initial_checkpoint_path
    if load_path:
        s = ckpt.restore(load_path)
        if expect_partial:
            s.expect_partial()

    if initial_checkpoint_path and not resuming_checkpoint_path:
        global_step_var.assign(0)





def main():
    init.initialize()   # FLAGS로 하이퍼파라미터 등 설정

    if FLAGS.train:
        train()
    elif FLAGS.predict:
        predict()
    elif FLAGS.export_file:
        export()


# this activates only when you explicitly call "main.py" file.
# Means that don't be activated when this is called by other files.
if __name__ == '__main__':
    main()

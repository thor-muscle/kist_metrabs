import functools
import glob
import xml
import xml.etree.ElementTree

import PIL
import cv2
import numpy as np

import improc
import paths
import util


@functools.lru_cache()
@util.cache_result_on_disk(
    f'{paths.DATA_ROOT}/cache/pascal_voc_occluders.pkl', min_time="2018-11-08T15:13:38")
def load_occluders(downscale_factor=0.5):
    image_mask_pairs = []
    pascal_root = f'{paths.DATA_ROOT}/pascal_voc'
    for annotation_path in glob.glob(f'{pascal_root}/Annotations/*.xml'):
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_person = (obj.find('name').text == 'person')
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        image_filename = xml_root.find('filename').text
        segmentation_filename = image_filename.replace('jpg', 'png')

        path = f'{pascal_root}/JPEGImages/{image_filename}'
        seg_path = f'{pascal_root}/SegmentationObject/{segmentation_filename}'

        im = improc.imread_jpeg(path)
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)
            object_image = im[ymin:ymax, xmin:xmax]
            # Ignore small objects
            if cv2.countNonZero(object_mask) < 500:
                continue

            object_mask = soften_mask(object_mask)
            object_image = improc.resize_by_factor(object_image, downscale_factor)
            object_mask = improc.resize_by_factor(object_mask, downscale_factor)
            image_mask_pairs.append((object_image, object_mask))

    return image_mask_pairs


def soften_mask(mask):
    morph_elem = improc.get_structuring_element(cv2.MORPH_ELLIPSE, (8, 8))
    eroded = cv2.erode(mask, morph_elem)
    result = mask.astype(np.float32)
    result[eroded < result] = 0.75
    return result

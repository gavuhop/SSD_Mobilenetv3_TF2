from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


def ssd_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
    '''
    Xây dựng model SSD300 với keras.
    Base network được sử dụng là VGG16.

    Chú ý: Yêu cầu Keras>=v2.0; TensorFlow backend>=v1.0.

    Arguments:
        image_size (tuple): Kích thước image input `(height, width, channels)`.
        n_classes (int): Số classes, chẳng hạn 20 cho Pascal VOC dataset, 80 cho MS COCO dataset.
        mode (str, optional): Một trong những dạng 'training', 'inference' và 'inference_fast'.
            'training' mode: Đầu ra của model là raw prediction tensor.
            'inference' và 'inference_fast' modes: raw predictions được decoded thành tọa độ đã được filtered thông qua threshold.
        l2_regularization (float, optional): L2-regularization rate. Áp dụng cho toàn bộ các convolutional layers.
        min_scale (float, optional): Nhân tố scaling nhỏ nhất cho các size của anchor boxes. Tỷ lệ này được tính trên so sánh với cạnh ngắn hơn
        của hình ảnh input.
        max_scale (float, optional): Nhân tố scale lớn nhất cho các size của anchor boxes.
        scales (list, optional): List các số floats chứa các nhân tố scaling của các convolutional predictor layer.
            List này phải lớn hơn số lượng các predictor layers là 1 để sử dụng cho trường hợp aspect ratio = 1 sẽ tính thêm next scale.
            Trong TH sử dụng scales thì interpolate theo min_scale và max_scale để tính list scales sẽ không được sử dụng.
        aspect_ratios_global (list, optional): List của các aspect ratios mà các anchor boxes được tạo thành. List này được áp dụng chung trên toàn bộ các prediction layers.
        aspect_ratios_per_layer (list, optional): List của các list aspect ratio cho mỗi một prediction layer.
            Nếu được truyền vào sẽ override `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Chỉ áp dụng khi aspect ratio lists chứa 1. Sẽ bị loại bỏ trong các TH khác.
            Nếu `True`, 2 anchor boxes sẽ được tạo ra ứng với aspect ratio = 1. anchor box đầu tiên tạo thành bằng cách sử scale, anchor box thứ 2
            được tạo thành bằng trung bình hình học của scale và next scale.
        steps (list, optional): `None` hoặc là list với rất nhiều các phần tử có số lượng bằng với số lượng layers.
            Mỗi phần tử đại diện cho mỗi một predictor layer có bao nhiêu pixels khoảng cách giữa các tâm của anchor box.
            steps có thể gồm 2 số đại diện cho (step_width, step_height).
            nếu không có steps nào được đưa ra thì chúng ta sẽ tính để cho khoảng các giữa các tâm của anchor box là bằng nhau
        offsets (list, optional): None hoặc là các con số đại diện cho mỗi một predictor layer bao nhiêu pixels từ góc trên và bên trái mở rộng của ảnh
        clip_boxes (bool, optional): Nếu `True`, giới hạn tọa độ các anchor box để nằm trong boundaries của image.
        variances (list, optional): Một list gồm 4 số floats >0. Một anchor box offset tương ứng với mỗi tọa độ sẽ được chi cho giá trị variance tương ứng.
        coords (str, optional): Tọa độ của box được sử dụng bên trong model (chẳng hạn, nó không là input format của ground truth labels).
            Có thể là dạng 'centroids' format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' format `(xmin, xmax, ymin, ymax)`, hoặc 'corners' format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Được đặt là `True` nếu model được giả định sử dụng tọa độ tương đối thay vì tuyệt đối coordinates,
            chẳng hạn nếu model dự báo tọa độ box nằm trong [0, 1] thay vì tọa độ tuyệt đối.
        subtract_mean (array-like, optional): `None` hoặc một array object với bất kì shape nào mà dạng mở rộng phù hợp với shape của ảnh. Gía trị của nó được bớt đi từ độ lớn pixel của ảnh. The elements of this array will be
            Chẳng hạn truyền vào một list gồm 3 số nguyên để tính toán trung bình chuẩn hóa cho các kênh của ảnh.
        divide_by_stddev (array-like, optional): `None` hoặc một array object. Tương tự như subtract_mean nhưng được chia cho từ độ lớn của ảnh để tính chuẩn hóa.
        swap_channels (list, optional): Là `False` hoặc một list các số nguyên biểu diễn thứ tự kì vọng mà trong đó đầu vào các channels của ảnh có thể được hoán đổi.
        confidence_thresh (float, optional): Một số float nằm trong khoảng [0,1), là ngưỡng tin cậy nhỏ nhất trong phân loại của một lớp xảy ra.
        iou_threshold (float, optional): Một float nằm trong khoảng [0,1]. Tất cả các boxes có chỉ số Jaccard similarity lớn hơn hoặc bằng `iou_threshold`
            sẽ được xem xét là chứa vệt thể bên trong nó.
        top_k (int, optional): Điểm dự báo cáo nhất được giữ trong mỗi batch item sau bước non-maximum suppression stage.
        nms_max_output_size (int, optional): Số lượng lớn nhất các dự báo sẽ được chuyển qua bước NMS stage.
        return_predictor_sizes (bool, optional): Nếu `True`, hàm số này sẽ không chỉ trả về mô hình, mà còn trả về
            một list chứa các chiều của predictor layers.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): Một numpy array chứa các phần `(height, width)` của output tensor shape tương ứng với mỗi convolutional predictor layer.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 6  # Số lượng các preductor convolutional layers trong network là 6 cho original SSD300.
    n_classes += 1  # Số lượng classes, + 1 để tính thêm background class.
    l2_reg = l2_regularization  # tham số chuẩn hóa của norm chuẩn l2.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Một số lỗi ngoại lệ.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    # Tạo list scales
    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Tính các tham số của anchor box.
    ############################################################################

    # Thiết lập aspect ratios cho mỗi predictor layer (chỉ cần thiết cho tính toán anchor box layers).
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Tính số lượng boxes được dự báo / 1 cell cho mỗi predictor layer.
    # Chúng ta cần biết bao nhiêu channels các predictor layers cần có.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 cho trường hợp aspect ratio = 1
            else:
                n_boxes.append(len(ar))
    else:  # Nếu chỉ 1 global aspect ratio list được truyền vào thì số lượng boxes là như nhau cho mọi layers.
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Xác định các hàm số cho Lambda layers bên dưới.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    ############################################################################
    # Bước 1: Xây dựng network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    ############################################################################
    # Bước 1.1: Tính toán base network là mạng VGG16
    ############################################################################

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    ############################################################################
    # Bước 1.2: Áp dụng các convolutional filter có kích thước (3 x 3) để tính toán ra features map.
    ############################################################################

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc6')(pool5)
    print('fully connected 6: ', fc6.get_shape())
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc7')(fc6)
    print('fully connected 7: ', fc7.get_shape())
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)
    print('conv6_2: ', conv6_2.get_shape())
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)
    print('conv7_2: ', conv7_2.get_shape())
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)
    print('conv8_2: ', conv8_2.get_shape())
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)
    print('conv9_2: ', conv9_2.get_shape())
    # Feed conv4_3 vào the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)
    print('conv4_3_norm.shape: ', conv4_3_norm.get_shape())

    ############################################################################
    # Bước 1.3: Xác định output phân phối xác suất theo các classes ứng với mỗi một default bounding box.
    ############################################################################

    ### Xây dựng các convolutional predictor layers tại top của base network
    # Chúng ta dự báo các giá trị confidence cho mỗi box, do đó confidence predictors có độ sâu `n_boxes * n_classes`
    # Đầu ra của confidence layers có shape: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    print('conv4_3_norm_mbox_conf.shape: ', conv4_3_norm_mbox_conf.get_shape())
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    print('fc7_mbox_conf.shape: ', fc7_mbox_conf.get_shape())
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
    print('conv9_2_mbox_conf: ', conv9_2_mbox_conf.get_shape())

    ############################################################################
    # Bước 1.4: Xác định output các tham số offset của default bounding boxes tương ứng với mỗi cell trên các features map.
    ############################################################################

    # Chúng ta dự báo 4 tọa độ cho mỗi box, do đó localization predictors có độ sâu `n_boxes * 4`
    # Output shape của localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    print('conv4_3_norm_mbox_loc: ', conv4_3_norm_mbox_loc.get_shape())
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)
    print('conv9_2_mbox_loc: ', conv9_2_mbox_loc.get_shape())

    ############################################################################
    # Bước 1.5: Tính toán các AnchorBoxes làm cơ sở để dự báo offsets cho các predicted bounding boxes bao quan vật thể
    ############################################################################

    ### Khởi tạo các anchor boxes (được gọi là "priors" trong code gốc Caffe/C++ của mô hình)
    # Shape output của anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    print('conv4_3_norm_mbox_priorbox: ', conv4_3_norm_mbox_priorbox.get_shape())
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='fc7_mbox_priorbox')(fc7_mbox_loc)
    print('fc7_mbox_priorbox: ', fc7_mbox_priorbox.get_shape())
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    print('conv6_2_mbox_priorbox: ', conv6_2_mbox_priorbox.get_shape())
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    print('conv7_2_mbox_priorbox: ', conv7_2_mbox_priorbox.get_shape())
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    print('conv8_2_mbox_priorbox: ', conv8_2_mbox_priorbox.get_shape())
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
    print('conv9_2_mbox_priorbox: ', conv9_2_mbox_priorbox.get_shape())

    ############################################################################
    # Bước 2: Reshape lại các output tensor shape
    ############################################################################

    ############################################################################
    # Bước 2.1: Reshape output của class predictions
    ############################################################################

    # Reshape các class predictions, trả về 3D tensors có shape `(batch, height * width * n_boxes, n_classes)`
    # Chúng ta muốn các classes là tách biệt nhau trên last axis để tính softmax trên chúng.
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
        conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    print('conv4_3_norm_mbox_conf_reshape: ', conv4_3_norm_mbox_conf_reshape.get_shape())
    print('fc7_mbox_conf_reshape: ', fc7_mbox_conf_reshape.get_shape())
    print('conv9_2_mbox_conf_reshape: ', conv9_2_mbox_conf_reshape.get_shape())
    print('conv9_2_mbox_conf_reshape: ', conv9_2_mbox_conf_reshape.get_shape())
    print('conv9_2_mbox_conf_reshape: ', conv9_2_mbox_conf_reshape.get_shape())

    ############################################################################
    # Bước 2.2: Reshape output của bounding box predictions
    ############################################################################

    # Reshape các box predictions, trả về 3D tensors có shape `(batch, height * width * n_boxes, 4)`
    # Chúng ta muốn 4 tọa độ box là tách biệt nhau trên last axis để tính hàm smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    print('conv4_3_norm_mbox_loc_reshape: ', conv4_3_norm_mbox_loc_reshape.get_shape())
    print('fc7_mbox_loc_reshape: ', fc7_mbox_loc_reshape.get_shape())
    print('conv6_2_mbox_loc_reshape: ', conv6_2_mbox_loc_reshape.get_shape())
    print('conv7_2_mbox_loc_reshape: ', conv7_2_mbox_loc_reshape.get_shape())
    print('conv8_2_mbox_loc_reshape: ', conv8_2_mbox_loc_reshape.get_shape())
    print('conv9_2_mbox_loc_reshape: ', conv9_2_mbox_loc_reshape.get_shape())

    ############################################################################
    # Bước 2.3: Reshape output của anchor box
    ############################################################################

    # Reshape anchor box tensors, trả về 3D tensors có shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
        conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)
    print('conv4_3_norm_mbox_priorbox_reshape: ', conv4_3_norm_mbox_priorbox_reshape.get_shape())
    print('fc7_mbox_priorbox_reshape: ', fc7_mbox_priorbox_reshape.get_shape())
    print('conv6_2_mbox_priorbox_reshape: ', conv6_2_mbox_priorbox_reshape.get_shape())
    print('conv7_2_mbox_priorbox_reshape: ', conv7_2_mbox_priorbox_reshape.get_shape())
    print('conv8_2_mbox_priorbox_reshape: ', conv8_2_mbox_priorbox_reshape.get_shape())
    print('conv9_2_mbox_priorbox_reshape: ', conv9_2_mbox_priorbox_reshape.get_shape())
    ### Concatenate các predictions từ các layers khác nhau

    ############################################################################
    # Bước 3: Concatenate các boxes trên layers
    ############################################################################

    ############################################################################
    # Bước 3.1: Concatenate confidence output box
    ############################################################################

    # Axis 0 (batch) và axis 2 (n_classes hoặc 4) là xác định duy nhất cho toàn bộ các predictions layer
    # nên chúng ta muốn concatenate theo axis 1, số lượng các boxes trên layer
    # Output shape của `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])
    print('mbox_conf.shape: ', mbox_conf.get_shape())

    ############################################################################
    # Bước 3.2: Concatenate location output box
    ############################################################################

    # Output shape của `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    print('mbox_loc.shape: ', mbox_loc.get_shape())

    ############################################################################
    # Bước 3.3: Concatenate anchor output box
    ############################################################################

    # Output shape của `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    print('mbox_priorbox.shape: ', mbox_priorbox.get_shape())

    ############################################################################
    # Bước 4: Tính toán output
    ############################################################################

    ############################################################################
    # Bước 4.1 : Xây dựng các hàm loss function cho confidence
    ############################################################################

    # tọa độ của box predictions sẽ được truyền vào hàm loss function,
    # nhưng cho các dự báo lớp, chúng ta sẽ áp dụng một hàm softmax activation layer đầu tiên
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate các class và box predictions và the anchors thành một large predictions vector
    # Đầu ra của `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])
    print('predictions.shape: ', predictions.get_shape())
    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                    fc7_mbox_conf._keras_shape[1:3],
                                    conv6_2_mbox_conf._keras_shape[1:3],
                                    conv7_2_mbox_conf._keras_shape[1:3],
                                    conv8_2_mbox_conf._keras_shape[1:3],
                                    conv9_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
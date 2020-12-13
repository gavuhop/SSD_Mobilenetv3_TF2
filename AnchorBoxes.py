from __future__ import division
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer

from bounding_box_utils.bounding_box_utils import convert_coordinates

class AnchorBoxes(Layer):
    '''
    Tác dụng: Tạo ra một output tensor chứa tọa độ của các anchor box và các biến thể dựa trên input tensor.
    Một tợp hợp các 2D anchor boxes được tạo ra dựa trên aspect ratios và scale trên mỗi một cells của grid cells. Các hộp được tham số hóa bằng các tọa độ `(xmin, xmax, ymin, ymax)`
    
    Input shape:
        4D tensor shape `(batch, channels, height, width)` nếu `dim_ordering = 'th'`
        or `(batch, height, width, channels)` nếu `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. 
        Chiều cuối cùng gồm 4 tọa độ của anchor box và 4 giá trị biến thể ở mỗi box.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):
        '''

        Arguments:
            img_height (int): chiều cao input images.
            img_width (int): chiều rộng input images.
            this_scale (float): một giá trị float thuộc [0, 1], nhân tố scaling kích thước để tạo các anchor boxes dựa trên một tỷ lệ so với cạnh ngắn hơn trong width và height.
            next_scale (float): giá trị tiếp theo của scale. Được thiết lập khi vào chỉ khi
                `self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): tợp hợp các aspect ratios của các default boxes được tạo ra từ layer này.
            two_boxes_for_ar1 (bool, optional): Được sử dụng chỉ khi `aspect_ratios` = 1.
                Nếu `True`, hai default boxes được tạo ra khi aspect ratio = 1. default box đầu tiên sử dụng scaling factor của layer tương ứng,
                default box thứ 2 sử dụng trung bình hình học giữa scaling factor và next scaling factor.
            clip_boxes (bool, optional): Nếu đúng `True`, giới hạn tọa độ anchor box nằm bên trong hình ảnh.
            variances (list, optional): Tợp hợp gồm 4 giá trị floats > 0. Là các anchor box offset tương ứng với mỗi tọa độ chia cho giá trị variances tương ứng của nó.
            coords (str, optional): Tọa độ của box được sử dụng trong model. Có thể là centroids định dạng `(cx, cy, w, h)` (tọa độ box center, width, height),
                hoặc 'corners' định dạng `(xmin, ymin, xmax,  ymax)`, hoặc 'minmax' định dạng `(xmin, xmax, ymin, ymax)`.
            normalize_coords (bool, optional): Nếu `True` mô hình sử dụng tọa độ tương đối thay vì tuyệt đối. Chẳng hạn mô hình dự đoán tọa độ nằm trong [0, 1] thay vì tọa độ tuyệt đối.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        # Tính toán số lượng boxes trên 1 cell. TH aspect ratios = 1 thì thêm 1 box.
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        Return: Trả về 1 anchor box tensor dựa trên shape của input tensor.

        Tensor này được thiết kế như là hằng số và không tham gia vào quá trình tính toán.

        Arguments:
            x (tensor): 4D tensor có shape `(batch, channels, height, width)` nếu `dim_ordering = 'th'`
                hoặc `(batch, height, width, channels)` nếu `dim_ordering = 'tf'`. Input cho layer này phải là output của các localization predictor layer.
        '''
        #####################################################
        # Bước 1: Tính toán with và heigth của box với mỗi aspect ratio
        #####################################################
        # Cạnh ngẵn hơn của hình ảnh có thể được sử dụng để tính `w` và `h` sử dụng `scale` và `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Tính toán box widths và heights cho toàn bộ aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Tính anchor box thông thường khi aspect ratio = 1.
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Tính version lớn hơn của anchor box sử dụng the geometric mean của scale và next scale.
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                # Trường hợp còn lại box_height = scale/sqrt(aspect ratio); box_width = scale*sqrt(aspect ratio)
                box_height = self.this_scale * size // np.sqrt(ar)
                box_width = int(self.this_scale * size * np.sqrt(ar))
                wh_list.append((box_width, box_height))
        # append vào width height list
        wh_list = np.array(wh_list)

        # Định hình input shape 
        if K.image_data_format() == 'tf':
            batch_size, feature_map_channels, feature_map_height, feature_map_width = x.get_shape().as_list()
        else:
            batch_size, feature_map_height, feature_map_width, feature_map_channels = x.get_shape().as_list()

        
        # Tính các center points của grid of box. Chúng là duy nhất đối với các aspect ratios.
        #####################################################
        # Bước 2: Tính các step size. Khoảng cách là bao xa giữa các anchor box center point theo chiều width và height.
        #####################################################
        if (self.this_steps is None):
            step_height = self.img_height // feature_map_height
            step_width = self.img_width // feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Tính toán các offsets cho anchor box center point đầu tiên từ góc trên cùng bên trái của hình ảnh.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        #####################################################
        # Bước 3: Tính toán các tọa độ của (cx, cy, w, h) theo tọa độ của image gốc.
        #####################################################
        # Bây h chúng ta có các offsets và step sizes, tính grid của anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) 
        cy_grid = np.expand_dims(cy_grid, -1) 
        

        # Tạo một 4D tensor có shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # Chiều cuối cùng sẽ chứa `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # đặt cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # đặt cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # đặt w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # đặt h
        
        # Chuyển `(cx, cy, w, h)` sang `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # Nếu `clip_boxes` = True, giới hạn các tọa độ nằm trên boundary của hình ảnh
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # Nếu `normalize_coords` = True, chuẩn hóa các tọa độ nằm trong khoảng [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        if self.coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        # Tạo một tensor chứa các variances và append vào `boxes_tensor`. 
        variances_tensor = np.zeros_like(boxes_tensor) # shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances # Mở rộng thêm variances
        # Bây h `boxes_tensor` trở thành tensor kích thước `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Bây h chuẩn bị trước một chiều cho `boxes_tensor` đại diện cho batch size và di chuyển copy theo chiều đó (theo kiểu lợp ngói, xem thêm np.tile)
        #  ta được một 5D tensor kích thước `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        if K.common.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else: 
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Test output of Anchor box
import tensorflow as tf
x = tf.random.normal(shape = (4, 38, 38, 512))

aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
two_boxes_for_ar1=True
steps=[8, 16, 32, 64, 100, 300]
offsets=None
clip_boxes=False
variances=[0.1, 0.1, 0.2, 0.2]
coords='centroids'
normalize_coords=True
subtract_mean=[123, 117, 104]
divide_by_stddev=None
swap_channels=[2, 1, 0]
confidence_thresh=0.01
iou_threshold=0.45
top_k=200
nms_max_output_size=400


# Thiết lập tham số
img_height = 300
img_width = 300 
img_channels = 3 
mean_color = [123, 117, 104] 
swap_channels = [2, 1, 0] 
n_classes = 20 
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] 
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True


anchors = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2])(x)
print('anchors shape: ', anchors.shape)


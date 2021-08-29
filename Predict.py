
from PIL import image
from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

im_fname = image.open("your detect imgae")
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()
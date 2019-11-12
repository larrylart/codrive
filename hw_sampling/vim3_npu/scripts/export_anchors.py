#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

IMG_INPUT_TENSOR = "import/normalized_input_image_tensor:0"
ANCHOR_BOXES_TENSOR = "import/anchors:0"
EXPECTED_ANCHORS = 1917

def code_gen(anchors):
  print("/**")
  print(" * Anchors")
  print(" *")
  print(" * SSD Anchor boxes for SSD/Mobilenet architectures")
  print(" * num_layers    = 6")
  print(" * min_scale     = 0.2")
  print(" * max_scale     = 0.95")
  print(" * aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]")
  print(" *")
  print(" * See: https://github.com/tensorflow/models/blob/master/research/object_detection/anchor_generators/multiple_grid_anchor_generator.py#L248")
  print(" */")
  print("#ifndef _NN_ANCHORS_H_")
  print("#define _NN_ANCHORS_H_")
  print("")
  print("#define NN_ANCHORS_NO	1917")
  print("")
  print("static float vect_anchors[] = {")


  for i in range(0, EXPECTED_ANCHORS):

    box_str = ",".join(["%.8f" % (pt) for pt in anchors[i, :]])
    print( box_str, end =",\n" )
    #all_str += ", ".join(["% .8f" % (pt) for pt in anchors[i, :]])
    #print("    arr[%d] = [ %s ]" % (i, box_str))

  print("};")
  print("")
  print("#endif")
  print("")
  pass


def main(_):
  if len(sys.argv) != 2:
    print("Must specify an input graph file! Usage: %s [Graph PB file]" % (sys.argv[0]))

  graph_file = sys.argv[1]
  with open(graph_file, 'rb') as f:
    serialized = f.read()

  tf.reset_default_graph()
  original_gdef = tf.GraphDef()
  original_gdef.ParseFromString(serialized)
  graph = tf.import_graph_def(original_gdef)

  img_placeholder = np.zeros((1, 300, 300, 3))

  with tf.Session(graph=graph) as sess:
    image_input_tensor = sess.graph.get_tensor_by_name(IMG_INPUT_TENSOR)
    anchors_tensor = sess.graph.get_tensor_by_name(ANCHOR_BOXES_TENSOR)

    anchors = sess.run(anchors_tensor, feed_dict = { image_input_tensor: img_placeholder })

  assert(anchors.shape == (EXPECTED_ANCHORS, 4))

  code_gen(anchors)

if __name__ == '__main__':
  tf.app.run()

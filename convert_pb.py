import tensorflow as tf
from create_tfrecord import *
from tensorflow.python.framework import graph_util


resize_width = 229
resize_height = 229
depths = 3
def freeze_graph_test(pb_path,image_path):

    with tf.Graph().as_default():
        out_put_graph_def = tf.GraphDef()
        with open(pb_path,"rb") as f:
            out_put_graph_def.ParseFromString(f.read())
            tf.import_graph_def(out_put_graph_def, name = "")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
            out_put_tensor_name = sess.graph.get_tensor_by_name("")

            im = read_image(image_path, resize_height, resize_width, normalization=True)
            im = im[np.newaxis,:]

            out = sess.run(out_put_tensor_name,feed_dict={
                input_image_tensor:im,
                input_keep_prob_tensor:1.0,
                input_is_training_tensor:False
            })
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name="score")
            class_id = tf.arg_max(score,1)
            print("pre class_id:{}".format(sess.run(class_id)))




def freeze_graph(input_checkpoint, output_graph):

    #指定输出的节点名称，该节点名称必须是原模型中存在的节点
    output_node_names = ""
    saver = tf.train.import_meta_graph(input_checkpoint+'.meta',clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(",")
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph."%len(output_graph_def.node))
        #for op in sess.graph.get_operations():
        #   print(op.name, op.values())

if __name__ == '__main__':
    input_checkpoint = 'models/model.ckpt-10000'

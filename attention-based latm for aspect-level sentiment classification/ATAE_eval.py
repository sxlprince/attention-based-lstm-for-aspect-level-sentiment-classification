import tensorflow as tf
import numpy as np
from utils import load_w2v, batch_index, load_word_embedding, load_aspect2id, load_inputs_twitter_at


x_raw = ["$T$ is always fresh and hot - ready to eat !", "food"]
y_test = [1]

word_id_mapping, w2v = load_word_embedding('data/restaurant/word_id_new.txt', 'data/restaurant/rest_2014_word_embedding_300_new.txt', 300)
# dict(3909)          3910 * 300
aspect_id_mapping, aspect_embed = load_aspect2id('data/restaurant/aspect_id_new.txt', word_id_mapping, w2v, 300)
# dict(1219)            1220 * 300
# print (aspect_id_mapping['food'])
# print ('sxlllllllllll')


def change_y_to_onehot(y):

    class_set = set([1,-1,0])
    n_class = 3
    y_onehot_mapping = {0: 0, 1: 1, -1: 2}
    #print (y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

def load_inputs_twitter_at(input_file, word_id_file, aspect_id_file,sentence_len):
    word_to_id = word_id_file
    print ('load word-to-id done!')
    aspect_to_id = aspect_id_file
    print ('load aspect-to-id done!')

    x= []

    aspect_words = []
    lines = input_file
    aspect_word = ' '.join(lines[1].lower().split())
    aspect_words.append(aspect_to_id.get(aspect_word, 0))
    sxl = change_y_to_onehot(y_test)
    words = lines[0].lower().split()
    ids = []
    for word in words:
        if word in word_to_id:
            ids.append(word_to_id[word])
        # ids = list(map(lambda word: word_to_id.get(word, 0), words))
        #print (len(sen_len))
    x.append(ids + [0] * (sentence_len - len(ids)))
    x = np.asarray(x, dtype=np.int32)
    print (sxl)
    return x, np.asarray(aspect_words)

a, b = load_inputs_twitter_at(x_raw, word_id_mapping, aspect_id_mapping,80)
print ('input:', a)
print ('aspect:', b)

#=================================================
checkpoint_file = tf.train.latest_checkpoint('E:/caffe/AI/deep learning/tensorflow/attention-based latm for aspect-level sentiment classification\models/logs/1531470805_-d1-1.0d2-1.0b-25r-0.01l2-0.001sen-80dim-300h-300c-3')
#checkpoint_file = tf.train.latest_checkpoint('')
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("inputs/x").outputs[0]
        print (input_x)
        aspect = graph.get_operation_by_name("inputs/aspect_id").outputs[0]
        print (aspect)
        sen_len = graph.get_operation_by_name("inputs/sen_len").outputs[0]
        keep_prob1_sxl = graph.get_operation_by_name("dropout_keep_prob1").outputs[0]
        keep_prob2_sxl = graph.get_operation_by_name("dropout_keep_prob2").outputs[0]

        #alpha = graph.get_operation_by_name("alphaa").outputs[0]
        losss = graph.get_operation_by_name("predict/predictions").outputs[0]

        result = sess.run(losss ,{input_x: a, aspect: b,sen_len:[11], keep_prob1_sxl:1.0, keep_prob2_sxl : 1.0})

print (result)
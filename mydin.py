#没做dice和正则
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Concatenate, Flatten
#准备数据  样本 10 * （2 + 3 + 1） * 5
#用户画像特征 dense
profile = tf.random_normal([10, 2], mean=1, stddev=0.1, dtype=tf.float32)
emb = tf.keras.layers.Embedding(1000, 5)
profile = emb(profile)
print(profile)  #(10, 2, 5)
#用户行为序列  sparse
behavior = tf.random_normal([10, 3], mean=1, stddev=0.1, dtype=tf.float32)
behavior = emb(behavior)
print(behavior)  #(10, 3, 5)
#target 
target = tf.random_normal([10, 1], mean=1, stddev=0.1, dtype=tf.float32)
target = emb(target)
print(target)  #(10, 1, 5)
#label
label = tf.constant([1,0,1,1,0,1,0,0,1,1], dtype=tf.float32)

#DIN
keys = behavior
queries = tf.keras.backend.repeat_elements(target, behavior.get_shape().as_list()[1], 1)  #(10, 3, 5)

att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1) #(10, 3, 20)
att_out = tf.layers.Dense( 5, activation='sigmoid')(att_input)  #(10, 3, 5)
att_out = tf.transpose(att_out, (0, 2, 1))  #(10, 5, 3)
att_out = tf.matmul(att_out, keys)   #(10, 5, 5)

deep_input_emb = Concatenate(1)([profile, att_out])  #(10, 7, 5)
deep_input_emb = Flatten()(deep_input_emb)  #(10, 35)

output = tf.layers.Dense(40, activation='dice')(deep_input_emb)  #dice未完成
output = tf.layers.Dense(20, activation='dice')(output)  #(10, 20)

final_logit = tf.layers.Dense(1, use_bias=False)(output)  #(10, 1)
x = tf.sigmoid(final_logit)
output = tf.reshape(x, (-1, 1))  #(10,1)
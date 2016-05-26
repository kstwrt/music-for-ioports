import tensorflow as tf
import wave
import numpy as np
import sys, subprocess, tempfile, struct

#parameters
lstm_size = 128
batch_size = 5
seq_len = 882
epochs = 10

sampleTime = 60 #seconds


# data loading
if len(sys.argv) < 2:
  print "usage: python %s <input.mp3>" % sys.argv[0]
  sys.exit(1)

mp3file = sys.argv[1]
wavfile = tempfile.NamedTemporaryFile('rb')
subprocess.call(['lame', '--decode', '--quiet', mp3file, wavfile.name])

w = wave.open(wavfile)
rawFrames = w.readframes(sampleTime * w.getframerate())
w.close()

interleaved = struct.unpack('h' * (len(rawFrames) / 2), rawFrames)

stereo = np.array(interleaved).reshape(len(interleaved) / 2, 2)

left = stereo[:,0].astype(np.float32)
scaleMin = left.min()
scaleMax = left.max()

left -= scaleMin
left /= scaleMax

input_sequences = left.reshape(-1, seq_len)
target_sequences = np.concatenate((left[1:],[0])).reshape(-1, seq_len)

instances = zip(input_sequences, target_sequences)


# tf training model graph
inputs = []
for i in range(seq_len):
  inputs.append(tf.placeholder(tf.float32, [batch_size, 1]))

embedding = tf.get_variable("embedding", [1, lstm_size])

emb_inputs = []
for i in range(seq_len):
  emb_input = tf.matmul(inputs[i], embedding)
  emb_inputs.append(emb_input)

targets = []
for i in range(seq_len):
  targets.append(tf.placeholder(tf.float32, [batch_size, 1]))

cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

(outputs, state) = tf.nn.rnn(cell, emb_inputs, dtype=tf.float32)

softmax_w = tf.get_variable("softmax_w", [lstm_size, 1])
softmax_b = tf.get_variable("softmax_b", [1])

logits = []
for i in range(seq_len):
  logits.append(tf.matmul(outputs[i], softmax_w) + softmax_b)

batch_losses = []
for i in range(seq_len):
  batch_losses.append(
    tf.reduce_mean((logits[i] - targets[i])**2)
  )
loss = tf.add_n(batch_losses) / seq_len

step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# tf sample model graph
sample_start = tf.random_uniform([1, 1])
sample_inp = tf.matmul(sample_start, embedding)
sample_state = cell.zero_state(1, tf.float32)
sample_out_list = []
for i in range(seq_len):
  if i > 0: tf.get_variable_scope().reuse_variables()
  sample_inp, sample_state = cell(sample_inp, sample_state)
  sample_out = tf.matmul(sample_inp, softmax_w) + softmax_b
  sample_out_list.append(sample_out)

sample_out = tf.concat(0, sample_out_list)


# tf run code
sess = tf.Session()
sess.run(tf.initialize_all_variables()) 

for ep in range(epochs):

  shuf = np.random.permutation(instances)

  for batch_idx in range(len(shuf) / batch_size):
    batch = shuf[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
    
    input_feed = {}
    for i in range(seq_len):
      input_feed[inputs[i].name] = batch[:,0,i:i+1]
      input_feed[targets[i].name] = batch[:,1,i:i+1]

    trainloss, s, _ = sess.run(
      [loss, sample_out, step],
      input_feed
    )

    map(lambda val: sys.stdout.write(str(val)+'\n'), s.flat)
    sys.stdout.flush()

    sys.stderr.write(str(trainloss)+'\n')
    sys.stderr.flush()

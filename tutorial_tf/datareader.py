import numpy as np

def get_ptb_dataset(dataset='train'):
  fn = 'ptb_data/ptb.{}.txt'
  for line in open(fn.format(dataset)):
    for word in line.split():
      yield word
    yield '<eos>'

def ptb_iterator(raw_data, batch_size, num_steps):
  '''

  :param raw_data:
  :param batch_size:
  :param num_steps:
  :return: shape (batch_size , num_steps) y中每个词是x中每个词的后一个词
  '''
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = data[:, i * num_steps:(i + 1) * num_steps]
    y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
    yield (x, y)
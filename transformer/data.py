from torch.autograd import Variable
from torchtext import data
from transformer.modules import subsequent_mask

from en_de_config import config

max_src_in_batch = config['batch_size']
max_tgt_in_batch = config['batch_size']

def batch_size_fn(new, count, size_so_far):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch
    global max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def rebatch(pad_idx, batch):
    """
    Fix order in torchtext to match ours
    """
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


class Batch(object):
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum().item()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
                     random_shuffler=None, shuffle=False, sort_within_batch=False):
                """Sort within buckets, then batch, then shuffle batches.
                Partitions data into chunks of size 100*batch_size, sorts examples within
                each chunk using sort_key, then batch these examples and shuffle the
                batches.
                This pool function was changed to deal with larger batches -> the batch_size_fn input was removed
                for p in data.batch(d, batch_size * 100, batch_size_fn):
                """
                for p in data.batch(d, batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=key), batch_size, batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

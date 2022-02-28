import time
import numpy as np
import paddle as P
from paddle.nn import functional as F
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification

pretrained = 'ernie-1.0'
init_checkpoint = 'xnli_cls_model/ckpt.bin'
max_seqlen = 128
place = P.CUDAPlace(0)
tokenizer = ErnieTokenizer.from_pretrained(pretrained)
model = ErnieModelForSequenceClassification.from_pretrained(
    pretrained, num_labels=3, name='')

if init_checkpoint is not None:
    sd = P.load(str(init_checkpoint))
    model.set_state_dict(sd)

cls = ["contradiction", "entailment", "neutral"]
bsz = 1
data = '他说，妈妈，我回来了。	校车把他放下后，他立即给他妈妈打了电话。'
acc = []
with P.no_grad():
    model.eval()
    segs = [tokenizer.encode(d) for d in data.split('\t')]
    seg_a, seg_b = tokenizer.truncate(segs[0][0], segs[1][0], seqlen=max_seqlen)
    sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
    start = time.time()
    sentence = P.to_tensor(np.expand_dims(sentence, 0))
    segments = P.to_tensor(np.expand_dims(segments, 0))
    print(sentence.shape, segments.shape)
    _, logits = model(sentence, segments)
    score = F.softmax(logits)
    end = time.time()
    import pdb
    pdb.set_trace()
    cur_pred = P.argmax(score, -1).numpy()
    print(cur_pred)
    print(logits, score)
    print(cls[logits.argmax(-1)])
    elaps = (end - start) * 1000
    print("[batch: {}] inference time = {}ms each_batch = {}ms".format(bsz, elaps, elaps / bsz))
    print('\n'.join(map(str, logits.numpy().tolist())))
    #a = (logits.argmax(-1) == label)

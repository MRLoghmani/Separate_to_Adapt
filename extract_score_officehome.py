from data import *
from utilities import *
from networks import *
import matplotlib.pyplot as plt
import numpy as np

num_known_classes = 65 #25
num_all_classes = 65

def skip(data, label, is_train):
    return False
batch_size = 32

def transform(data, label, is_train):
    label = one_hot(num_all_classes,label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

ds = FileListDataset('/mnt/datasets/office-home/product_0-64_val.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
product = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)

ds = FileListDataset('/mnt/datasets/office-home/real_world_0-64_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
real_world = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)

ds = FileListDataset('/mnt/datasets/office-home/art_0-64_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
art = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)

ds = FileListDataset('/mnt/datasets/office-home/clipart_0-64_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
clipart = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)

setGPU('0')

discriminator_p = Discriminator(n = 25).cuda() # multi-binary classifier
discriminator_p.load_state_dict(torch.load('discriminator_p_office-home.pkl'))
feature_extractor = ResNetFc(model_name='resnet50')
cls = CLS(feature_extractor.output_num(), num_known_classes+1, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls).cuda()

score_pr = []
score_rw = []
score_ar = []
score_cl = []
label_pr = []
label_rw = []
label_ar = []
label_cl = []

def get_score(dataset):
    ss = []
    ll = []
    for (i, (im, label)) in enumerate(dataset.generator()):
        im = Variable(torch.from_numpy(im)).cuda()
        f, __, __, __ = net.forward(im)
        p = discriminator_p.forward(f).cpu().detach().numpy()
        ss.append(p)
        ll.append(label)
    return np.vstack(ss), np.vstack(ll)


score_pr, label_pr = get_score(product)
score_rw, label_rw = get_score(real_world)
score_ar, label_ar = get_score(art)
score_cl, label_cl = get_score(clipart)

filename = "scores_office-home"
np.savez_compressed(filename, 
                    product_score=score_pr, product_label=label_pr, 
                    real_world_score=score_rw, real_world_label=label_rw, 
                    art_score=score_ar, art_label=label_ar, 
                    clipart_score=score_cl, clipart_label=label_cl)



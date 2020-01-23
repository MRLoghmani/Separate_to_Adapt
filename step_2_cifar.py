from data import *
from utilities import *
from networks_cifar import *
import matplotlib.pyplot as plt
import numpy as np
import sys

from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10
from torchvision import transforms

def skip(data, label, is_train):
    return False
batch_size = int(sys.argv[1]) #32
learning_rate = float(sys.argv[2]) / 2.0 #5e-4
rank_interval = int(batch_size / 16)

def get_dataset():
    train_dataset = CIFAR10('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

    test_dataset = STL10('../data', split='train', download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

    def map_labels_stl_to_cifar10(labels_stl):
        # map labels...
        #   2 -> 1, 1 -> 2
        #   6 -> 7, 7 -> 6

        new_labels = np.zeros(len(labels_stl))
        for idx, l in enumerate(labels_stl):
            if l == 1:
                new_labels[idx] = 2
            elif l == 2:
                new_labels[idx] = 1
            elif l >= 5:
                new_labels[idx] = 5
            else:
                new_labels[idx] = l

        return np.asarray(new_labels)

    def apply_one_hot(labels):
        new_labels = np.zeros(len(labels))
        for idx, l in enumerate(labels):
            if l >= 5:
                new_labels[idx] = 5
            else:
                new_labels[idx] = l

        return np.asarray(new_labels)

    test_dataset.labels = map_labels_stl_to_cifar10(test_dataset.labels)
    train_dataset.train_labels = apply_one_hot(train_dataset.train_labels)
    
    return train_dataset, test_dataset

source_dataset, target_dataset = get_dataset()

source_loader = torch.utils.data.DataLoader(source_dataset, 
    batch_size=batch_size, shuffle=True, num_workers=0)

target_loader = torch.utils.data.DataLoader(target_dataset,
    batch_size=batch_size, shuffle=True, num_workers=0)

setGPU('0')
log = Logger('log/Step_2', clear=True)


#discriminator_t = CLS_0(2048,2,bottle_neck_dim = 256).cuda()
discriminator_t = CLS_0(100,2,bottle_neck_dim = 100).cuda()
#----------------------------load the known/unknown discriminator
discriminator_t.load_state_dict(torch.load('discriminator_a_cifar.pkl'))
discriminator = SmallAdversarialNetwork(100).cuda()
#feature_extractor = ResNetFc(model_name='resnet50',model_path='/home/youkaichao/data/pytorchModels/resnet50.pth')
feature_extractor = CIFAR10Fc()
feature_extractor.load_state_dict(torch.load('feature_extractor_a_cifar.pkl'))
cls = CLS(feature_extractor.output_num(), 6, bottle_neck_dim=100)
net = nn.Sequential(feature_extractor, cls).cuda()

scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)

optimizer_discriminator = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_feature_extractor = OptimWithSheduler(optim.SGD(feature_extractor.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

# =========================weighted adaptation of the source and target domains
print("=========================weighted adaptation of the source and target domains")                            
k=0
while k <1500:
    for i, (batch_source, batch_target) in enumerate(zip(source_loader, target_loader)):

        im_source, label_source = batch_source
        im_target, label_target = batch_target

        def one_hot_encoding(y):
            y_onehot = y.numpy()
            y_onehot = (np.arange(6) == y_onehot[:,None]).astype(np.float32)
            y_onehot = torch.from_numpy(y_onehot)
            return y_onehot

        label_source = one_hot_encoding(label_source)
        label_target = one_hot_encoding(label_target)

        im_source, label_source = im_source.cuda(), label_source.cuda(non_blocking=True)
        im_target, label_target = im_target.cuda(), label_target.cuda(non_blocking=True)
         
        _, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        
        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
        
        try:
            __,_,_,dptarget = discriminator_t.forward(ft1.detach())
            r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][batch_size-rank_interval:]
            feature_otherep = torch.index_select(ft1, 0, r.view(rank_interval))
            _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
            ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((rank_interval,5)), np.ones((rank_interval,1))), axis = -1).astype('float32'))).cuda(),predict_prob_otherep)
        except:
            continue

        ce = CrossEntropyLoss(label_source, predict_prob_source)

        entropy = EntropyLoss(predict_prob_target, instance_level_weight= dptarget[:,0].contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), predict_prob=domain_prob_discriminator_1_source )
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), predict_prob=1 - domain_prob_discriminator_1_target, 
                                      instance_level_weight = dptarget[:,0].contiguous())

        with OptimizerManager([optimizer_cls, optimizer_feature_extractor,optimizer_discriminator]):
            loss = ce + 0.3 * adv_loss + 0.1 * entropy 
            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 10 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train', 'adv_loss','entropy','ce_ep'], globals())

        if log.step % 100 == 0:
            clear_output()
            

# =========================eliminate unknown samples 
print("=========================eliminate unknown samples")
k=0
while k <400:
    for i, (batch_source, batch_target) in enumerate(zip(source_loader, target_loader)):
        im_source, label_source = batch_source
        im_target, label_target = batch_target

        def one_hot_encoding(y):
            y_onehot = y.numpy()
            y_onehot = (np.arange(6) == y_onehot[:,None]).astype(np.float32)
            y_onehot = torch.from_numpy(y_onehot)
            return y_onehot

        label_source = one_hot_encoding(label_source)
        label_target = one_hot_encoding(label_target)

        im_source, label_source = im_source.cuda(), label_source.cuda(non_blocking=True)
        im_target, label_target = im_target.cuda(), label_target.cuda(non_blocking=True)
         
        _, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        
        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
        
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())
        r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][batch_size-rank_interval:]

        try:
            feature_otherep = torch.index_select(ft1, 0, r.view(rank_interval))
        except:
            continue
        _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((rank_interval,5)), np.ones((rank_interval,1))), axis = -1).astype('float32'))).cuda(),predict_prob_otherep)

        ce = CrossEntropyLoss(label_source, predict_prob_source)

        entropy = EntropyLoss(predict_prob_target, instance_level_weight= dptarget[:,0].contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), predict_prob=domain_prob_discriminator_1_source )
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), predict_prob=1 - domain_prob_discriminator_1_target, 
                                      instance_level_weight = dptarget[:,0].contiguous())

        with OptimizerManager([optimizer_cls, optimizer_feature_extractor,optimizer_discriminator]):
            loss = ce + 0.3 * adv_loss + 0.1 * entropy + 0.3 * ce_ep
            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 10 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train', 'adv_loss','entropy','ce_ep'], globals())

        if log.step % 100 == 0:
            clear_output()
            
torch.cuda.empty_cache()

# =================================evaluation
print("# =================================evaluation")

with TrainingModeManager([feature_extractor,discriminator_t, cls], train=False) as mgr, Accumulator(['predict_prob','dp','predict_index', 'label']) as accumulator:
    for i, batch_target in enumerate(target_loader):
        im, label = batch_target

        def one_hot_encoding(y):
            y_onehot = y.numpy()
            y_onehot = (np.arange(6) == y_onehot[:,None]).astype(np.float32)
            y_onehot = torch.from_numpy(y_onehot)
            return y_onehot

        label = one_hot_encoding(label)
        im, label = im.cuda(), label.cuda(non_blocking=True)

        ss, fs,_,  predict_prob = net.forward(im)
        _,_,_,dp = discriminator_t.forward(ss)
        predict_prob, dp,label = [variable_to_numpy(x) for x in (predict_prob,dp[:,1], label)]
        label = np.argmax(label, axis=-1).reshape(-1, 1)
        predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
        accumulator.updateData(globals())

        if i % 10 == 0:
            print(i)
        

for x in accumulator.keys():
    globals()[x] = accumulator[x]

y_true = label.flatten()
y_pred = predict_index.flatten()
m = extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=list(np.arange(11)))

cm = m
cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
acc_os_star = sum([cm[i][i] for i in range(5)]) / 5
acc_unk = cm[5][5]
acc_os = (acc_os_star * 5 + acc_unk) / 6
acc_all = sum([cm[i][i] for i in range(6)]) / 6

print("OS = {}, OS* = {}, UNK = {}, ALL = {}".format(acc_os, acc_os_star, acc_unk, acc_all))

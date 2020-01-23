from data import *
from utilities import *
from networks_mnistm import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from torchvision.utils import save_image

from torchvision.datasets import MNIST
from mnistm import *
from torchvision import transforms

def skip(data, label, is_train):
    return False
batch_size = int(sys.argv[1]) #32
learning_rate = float(sys.argv[2]) #1e-3
rank_interval = int(batch_size / 16)

def get_dataset():
    train_dataset = MNISTM('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
    
    test_dataset = MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

    image_path = []
    image_label = []
    for i in range(len(train_dataset.train_data)):
        if int(train_dataset.train_labels[i]) < 5:
            image_path.append(train_dataset.train_data[i])
            image_label.append(train_dataset.train_labels[i])
    train_dataset.train_data = image_path
    train_dataset.train_labels = image_label
    
    return train_dataset, test_dataset

source_dataset, target_dataset = get_dataset()

source_loader = torch.utils.data.DataLoader(source_dataset, 
    batch_size=batch_size, shuffle=True, num_workers=0)

target_loader = torch.utils.data.DataLoader(target_dataset,
    batch_size=batch_size, shuffle=True, num_workers=0)


setGPU('0')
log = Logger('log/step_1', clear=True)

#discriminator_t = CLS_0(2048,2,bottle_neck_dim = 256).cuda()
discriminator_t = CLS_0(500, 2, bottle_neck_dim=100).cuda()
discriminator_p = Discriminator(n = 5).cuda()
#feature_extractor = ResNetFc(model_name='resnet50',model_path='/home/liuhong/data/pytorchModels/resnet50.pth')
feature_extractor = MNISTFc()
cls = CLS(feature_extractor.output_num(), 6, bottle_neck_dim=100)
net = nn.Sequential(feature_extractor, cls).cuda()

scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_discriminator_t = OptimWithSheduler(optim.SGD(discriminator_t.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_discriminator_p = OptimWithSheduler(optim.SGD(discriminator_p.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
#optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),
#                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_net = OptimWithSheduler(optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4),
                            scheduler)

# =========================train feature extractor
k=0
while k < 10000:
    for i, batch_source in enumerate(source_loader):

        im_source, label_source = batch_source

        def one_hot_encoding(y):
            y_onehot = y.numpy()
            y_onehot = (np.arange(6) == y_onehot[:,None]).astype(np.float32)
            y_onehot = torch.from_numpy(y_onehot)
            return y_onehot

        label_source = one_hot_encoding(label_source)


        if i == 0:
            for j in range(5):
                save_image(im_source[j], "./"+str(j)+".png")
                print(label_source[j])

        im_source, label_source = im_source.cuda(), label_source.cuda(non_blocking=True)
        fs1, feature_source, __, predict_prob_source = net.forward(im_source)

        # =========================loss function
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        
        with OptimizerManager([optimizer_net]):
            loss = ce  
            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 1000 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train'], globals())

        if log.step % 5000 == 0:
            clear_output()

# =========================train the multi-binary classifier
k=0
while k < 500:
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
        
        fs1, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        

        p0 = discriminator_p.forward(fs1)
        p1 = discriminator_p.forward(ft1)
        p2 = torch.sum(p1, dim = -1)
     
        # =========================rank the output of the multi-binary classifiers
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())

        try:
            r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][batch_size-rank_interval:]
            feature_otherep = torch.index_select(ft1, 0, r.view(rank_interval))
            _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
            w = torch.sort(p2.detach(),dim = 0)[1][batch_size-rank_interval:]
            h = torch.sort(p2.detach(),dim = 0)[1][0:rank_interval]
            feature_otherep2 = torch.index_select(ft1, 0, w.view(rank_interval))
            feature_otherep1 = torch.index_select(ft1, 0, h.view(rank_interval))
            _,_,_,pred00 = discriminator_t.forward(feature_otherep2)
            _,_,_,pred01 = discriminator_t.forward(feature_otherep1)
        except:
            continue

        # =========================loss function
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        d1 = BCELossForMultiClassification(label_source[:,0:5],p0)
        
        with OptimizerManager([optimizer_cls, optimizer_discriminator_p]):
            loss = ce + d1  
            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 10 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train', 'd1'], globals())

        if log.step % 100 == 0:
            clear_output()
            
# =========================train the known/unknown discriminator
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
        
        
        fs1, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        
        p0 = discriminator_p.forward(fs1)
        p1 = discriminator_p.forward(ft1)
        p2 = torch.sum(p1, dim = -1)
     
        # =========================rank the output of the multi-binary classifiers

        try:
            __,_,_,dptarget = discriminator_t.forward(ft1.detach())
            r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][batch_size-rank_interval:]
            feature_otherep = torch.index_select(ft1, 0, r.view(rank_interval))
            _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
            w = torch.sort(p2.detach(),dim = 0)[1][batch_size-rank_interval:]
            h = torch.sort(p2.detach(),dim = 0)[1][0:rank_interval]
            feature_otherep2 = torch.index_select(ft1, 0, w.view(rank_interval))
            feature_otherep1 = torch.index_select(ft1, 0, h.view(rank_interval))
            _,_,_,pred00 = discriminator_t.forward(feature_otherep2)
            _,_,_,pred01 = discriminator_t.forward(feature_otherep1)
        except:
            continue

        # =========================loss function
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        d1 = BCELossForMultiClassification(label_source[:,0:5],p0)
        d2 = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.ones((rank_interval,1)), np.zeros((rank_interval,1))), axis = -1).astype('float32'))).cuda(),pred00)
        d2 += CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((rank_interval,1)), np.ones((rank_interval,1))), axis = -1).astype('float32'))).cuda(),pred01)
        
        with OptimizerManager([optimizer_cls, optimizer_discriminator_p, optimizer_discriminator_t]):
            loss = ce + d1 +d2 
            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 10 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train', 'd1', 'd2'], globals())

        if log.step % 100 == 0:
            clear_output()

# =========================save the parameters of the known/unknown discriminator
torch.save(discriminator_t.state_dict(), 'discriminator_a_mnist.pkl')
torch.save(feature_extractor.state_dict(), 'feature_extractor_a_mnist.pkl')


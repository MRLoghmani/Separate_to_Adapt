from data import *
from utilities import *
from networks import *
import matplotlib.pyplot as plt
import numpy as np
import sys

source_ds = sys.argv[1]
target_ds = sys.argv[2]
num_known_classes = int(sys.argv[3])
id_string = sys.argv[4] #'10-14'
num_exper = sys.argv[5]
gpu_id = sys.argv[6]

num_all_classes = 65

def skip(data, label, is_train):
    return False
batch_size = 32

def transform(data, label, is_train):
    label = one_hot(num_known_classes+1, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

#ds = FileListDataset('/mnt/datasets/office-home/split_files/'+source_ds+'_0-'+str(num_known_classes-1)+'_train.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
ds = FileListDataset('/mnt/datasets/office-home/split_files/'+source_ds+'_'+id_string+'_train.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)

source_train = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)

def transform(data, label, is_train):
    if label in range(num_known_classes):
        label = one_hot(num_known_classes+1, label)
    else:
        label = one_hot(num_known_classes+1,num_known_classes)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

#ds1 = FileListDataset('/mnt/datasets/office-home/split_files/'+target_ds+'_0-64_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)

#ds1 = FileListDataset('/mnt/datasets/office-home/split_files/'+target_ds+'_compl_'+id_string+'_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)

ds1 = FileListDataset('/mnt/datasets/office-home/split_files/'+target_ds+'_0-64_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=True, imsize=256)

target_train = CustomDataLoader(ds1, batch_size=batch_size, num_threads=2)

def transform(data, label, is_train):
    label = one_hot(num_all_classes,label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

#ds2 = FileListDataset('/mnt/datasets/office-home/split_files/'+target_ds+'_0-64_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=False, imsize=256)

ds2 = FileListDataset('/mnt/datasets/office-home/split_files/'+target_ds+'_0-64_test.txt', '/mnt/datasets/office-home/', transform=transform, skip_pred=skip, is_train=False, imsize=256)

target_test = CustomDataLoader(ds2, batch_size=batch_size, num_threads=2)

setGPU(gpu_id)
log = Logger('log/step_1', clear=True)

discriminator_t = CLS_0(2048,2,bottle_neck_dim = 256).cuda() # known/unknown discriminator
discriminator_p = Discriminator(n = num_known_classes).cuda() # multi-binary classifier
feature_extractor = ResNetFc(model_name='resnet50')
cls = CLS(feature_extractor.output_num(), num_known_classes+1, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls).cuda()

scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_discriminator_t = OptimWithSheduler(optim.SGD(discriminator_t.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_discriminator_p = OptimWithSheduler(optim.SGD(discriminator_p.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

# =========================train the multi-binary classifier
k=0
while k <500:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):

        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
        
        fs1, feature_source, _, predict_prob_source = net.forward(im_source)
        ft1, feature_target, _, predict_prob_target = net.forward(im_target)

        p0 = discriminator_p.forward(fs1)
        p1 = discriminator_p.forward(ft1)
        #p2 = torch.sum(p1, dim = -1)
        p2 = torch.max(p1, dim = -1)[0]

        # =========================rank the output of the multi-binary classifiers
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())
        r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][30:]
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
        w = torch.sort(p2.detach(),dim = 0)[1][30:]
        h = torch.sort(p2.detach(),dim = 0)[1][0:2]
        #w = torch.sort(p2[0].detach(),dim = 0)[1][30:]
        #h = torch.sort(p2[0].detach(),dim = 0)[1][0:2]

        feature_otherep2 = torch.index_select(ft1, 0, w.view(2))
        feature_otherep1 = torch.index_select(ft1, 0, h.view(2))
        _,_,_,pred00 = discriminator_t.forward(feature_otherep2)
        _,_,_,pred01 = discriminator_t.forward(feature_otherep1)

        # =========================loss function
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        d1 = BCELossForMultiClassification(label_source[:,0:num_known_classes],p0)
        
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
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):

        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
        
        fs1, feature_source, _, predict_prob_source = net.forward(im_source)
        ft1, feature_target, _, predict_prob_target = net.forward(im_target)
        
        p0 = discriminator_p.forward(fs1)
        p1 = discriminator_p.forward(ft1)
        #p2 = torch.sum(p1, dim = -1)
        p2 = torch.max(p1, dim = -1)[0]
     
        # =========================rank the output of the multi-binary classifiers
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())
        r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][30:]
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
        w = torch.sort(p2.detach(),dim = 0)[1][30:]
        h = torch.sort(p2.detach(),dim = 0)[1][0:2]
        #w = torch.sort(p2[0].detach(),dim = 0)[1][30:]
        #h = torch.sort(p2[0].detach(),dim = 0)[1][0:2]

        feature_otherep2 = torch.index_select(ft1, 0, w.view(2))
        feature_otherep1 = torch.index_select(ft1, 0, h.view(2))
        _,_,_,pred00 = discriminator_t.forward(feature_otherep2)
        _,_,_,pred01 = discriminator_t.forward(feature_otherep1)

        # =========================loss function
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        d1 = BCELossForMultiClassification(label_source[:,0:num_known_classes],p0)
        d2 = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.ones((2,1)), np.zeros((2,1))), axis = -1).astype('float32'))).cuda(),pred00)
        d2 += CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,1)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(),pred01)
        
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
torch.save(discriminator_p.state_dict(), 'discriminator_p_'+id_string+'_known_office-home_'+source_ds+'_'+target_ds+'_'+num_exper+'.pkl')
torch.save(discriminator_t.state_dict(), 'discriminator_t_'+id_string+'_known_office-home_'+source_ds+'_'+target_ds+'_'+num_exper+'.pkl')


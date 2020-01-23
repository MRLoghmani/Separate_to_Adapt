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

#num_known_classes = 10
num_all_classes = 31

def skip(data, label, is_train):
    return False
batch_size = 32

def transform(data, label, is_train):
    label = one_hot(num_known_classes+1, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
ds = FileListDataset('/mnt/datasets/office-31/amazon/images/known_split.txt', '/mnt/datasets/office-31/amazon/images/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
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
ds1 = FileListDataset('/mnt/datasets/office-31/webcam/images/os_split.txt', '/mnt/datasets/office-31/webcam/images/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
target_train = CustomDataLoader(ds1, batch_size=batch_size, num_threads=2)

def transform(data, label, is_train):
    label = one_hot(num_all_classes, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
ds2 = FileListDataset('/mnt/datasets/office-31/webcam/images/os_split.txt', '/mnt/datasets/office-31/webcam/images/', transform=transform, skip_pred=skip, is_train=False, imsize=256)
target_test = CustomDataLoader(ds2, batch_size=batch_size, num_threads=2)

setGPU(gpu_id)
log = Logger('log/Step_2', clear=True)


discriminator_t = CLS_0(2048,2,bottle_neck_dim = 256).cuda()
#discriminator_t = CLS_0(4096,2,bottle_neck_dim = 256).cuda()
#----------------------------load the known/unknown discriminator
discriminator_t.load_state_dict(torch.load('discriminator_a_office31'+source_ds+'_'+target_ds+'_'+num_exper+'.pkl'))
discriminator = LargeAdversarialNetwork(256).cuda()
feature_extractor = ResNetFc(model_name='resnet50')
#feature_extractor = AlexNetFc()
cls = CLS(feature_extractor.output_num(), num_known_classes+1, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls).cuda()

scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)

optimizer_discriminator = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_feature_extractor = OptimWithSheduler(optim.SGD(feature_extractor.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

# =========================weighted adaptation of the source and target domains                            
k=0
while k <1500:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):
        
        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
         
        _, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        
        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
        
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())
        r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][30:]
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,num_known_classes)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(),predict_prob_otherep)
        
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
k=0
while k <400:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):
        
        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
         
        _, feature_source, __, predict_prob_source = net.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net.forward(im_target)
        
        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
        
        __,_,_,dptarget = discriminator_t.forward(ft1.detach())
        r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][30:]
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, _, __, predict_prob_otherep = cls.forward(feature_otherep)
        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,num_known_classes)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(),predict_prob_otherep)
        
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
with TrainingModeManager([feature_extractor,discriminator_t, cls], train=False) as mgr, Accumulator(['predict_prob','dp','predict_index', 'label']) as accumulator:
    for (i, (im, label)) in enumerate(target_test.generator()):

        im = Variable(torch.from_numpy(im), volatile=True).cuda()
        label = Variable(torch.from_numpy(label), volatile=True).cuda()
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
m = extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=list(np.arange(num_known_classes+1)))

cm = m
cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
acc_os_star = np.mean([cm[i][i] for i in range(num_known_classes)])
acc_ukw = np.mean([cm[i][num_known_classes] for i in range(num_known_classes,cm.shape[0])])
acc_os = (acc_os_star * num_known_classes + acc_ukw) / (num_known_classes+1)
hos = 2*acc_os_star*acc_ukw / (acc_os_star+acc_ukw)
print("OS*={} , UKW={} , OS={} , HOS={}".format(acc_os_star, acc_ukw, acc_os, hos))
result=str(acc_os_star)+" "+str(acc_ukw)+" "+str(acc_os)+" "+str(hos)

with open(source_ds+'_'+target_ds+'_'+num_exper+'.txt', 'w+') as f:
    f.write("OS={} , OS*={} , UKW={} , HOS={}".format(acc_os, acc_os_star, acc_ukw, hos))

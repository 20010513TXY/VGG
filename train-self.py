import json
import os.path
from time import time

import torch
from thop import profile, clever_format
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import vgg

writer = SummaryWriter('logs')

def main():
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')
    print("Using device:{}".format(device))
    # 数据预处理
    data_transform = {
        "train" : transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
        "val" : transforms.Compose([transforms.Compose([transforms.RandomResizedCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])]),
    }

    data_root = r'E:\研\研究生\code\AlexNet\flower_data'
    assert os.path.exists(data_root),"{} path does  not exsit!".format(data_root)
    train_datasets = datasets.ImageFolder(os.path.join(data_root,'train'),
                                          transform=data_transform['train'])
    val_datasets = datasets.ImageFolder(os.path.join(data_root,'val'),
                                        transform=data_transform['val'])
    traindata_num = len(train_datasets)
    valdata_num = len(val_datasets)
    print("训练集数据:{},验证集数据:{}".format(traindata_num,valdata_num))
    flower_list = train_datasets.class_to_idx
    cla_dic = dict((val,key) for key,val in flower_list.items())
    json_str = json.dumps(cla_dic,indent=4)
    with open('class_indice.json','w') as json_file:
        json_file.write(json_str)
    batch_size = 32
    nw = min(os.cpu_count(),batch_size if batch_size > 1 else 0,8)
    print("the num_work is :{}".format(nw))

    train_dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=batch_size,
                                                   shuffle = True,num_workers = nw)
    val_dataloader = torch.utils.data.DataLoader(val_datasets,batch_size=batch_size,
                                                 shuffle = True,num_workers = nw)

    # 模型 损失函数和优化函数
    model_name = 'vgg16'
    net = vgg(model_name,num_classes = 5,init_weights = True)
    net.to(device)
    # flops,params = profile(net,inputs = (torch.zeros(1,3,224,224).to(device),))
    # flops,params = clever_format([flops,params],"%.3f")
    # print("vgg模型的计算量为:{},参数为:{}".format(flops,params))
    #vgg模型的计算量为:15.466G,参数为:134.281M
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = './{}_Net.pth'.format(model_name)

    for epoch in range(epochs):
        # 开始训练
        net.train()
        running_loss = 0.0
        time_A = time()
        for train_data in train_dataloader:
            images,labels = train_data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs,labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        time_B = time()
        time_C = time_B - time_A
        print("训练时,一个epoch所花费时间为:{}".format(time_C))
        print("训练时:第{}个epoch的损失为:{}".format(epoch+1,running_loss))
        writer.add_scalar('train_loss',running_loss,epoch)

        # 开始验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            time_A = time()
            for val_data in val_dataloader:
                val_images,val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                #print(predict_y)
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item()
            time_B = time()
            time_C = time_B - time_A
            print("验证时,一个epoch所花费时间为:{}".format(time_C))
            acc_rate = acc/valdata_num
            print("验证时:第{}个epoch的准确率是{}".format(epoch+1,acc_rate))
            writer.add_scalar("val_acc_rate",acc_rate,epoch)
        if acc_rate > best_acc:
            best_acc = acc_rate
            torch.save(net.state_dict(),save_path)

    print("Finished!")


if __name__ == '__main__':
    main()
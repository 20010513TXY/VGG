import json
import os.path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from model import vgg


def main():
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
    print("Using device:{}".format(device))

    #数据预处理
    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    img_path = r'向日葵.jpg'
    assert os.path.exists(img_path),"Image {} is not exsit!".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img,dim=0)

    # 读取class_indice.json文件
    json_path = 'class_indice.json'
    assert os.path.exists(json_path),"file {} is not exist!".format(json_path)
    with open(json_path,'r') as f:
        class_indict = json.load(f)

    # 加载模型
    model = vgg(model_name='vgg16',num_classes = 5).to(device)
    weights_path = 'vgg16_Net.pth'
    assert os.path.exists(weights_path),"file {} is not exist!".format(weights_path)
    model.load_state_dict(torch.load(weights_path,map_location=device))

    # 测试图片
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output,dim=0)
        predict_y = torch.argmax(predict).numpy()

    print_res = "class : {}   prob:{:.3}".format(class_indict[str(predict_y)],predict[predict_y].numpy())
    plt.title(print_res)
    for i in  range(len(predict)):
        print("class:{:10}   prob:{:.3}".format(class_indict[str(i)],predict[i].numpy()))
    plt.show()

if __name__ == "__main__":
    main()
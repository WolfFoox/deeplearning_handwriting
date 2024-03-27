import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import swin_tiny_patch4_window7_224 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    T=0

    for i in range(20):
        if i==0:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/BQTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==1:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/DJTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==2:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/DJMTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==3:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/HHRTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==4:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/HYFTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==5:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/JLCTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==6:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/LDTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==7:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/LLLTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==8:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/LQYTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==9:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/LXYTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==10:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/LYTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==11:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/LZGTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==12:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/MHTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==13:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/MTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==14:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/QZHTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==15:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/TZLTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==16:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/WJMTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==17:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/WJYTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==18:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/YXNTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]
        elif i==19:
            # load image
            # 指向需要遍历预测的图像文件夹
            imgs_root = "../Test/ZMTEST"
            assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
            # 读取指定文件夹下所有jpg图像路径
            img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]





        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = create_model(num_classes=79).to(device)
        # load model weights
        model_weight_path = "./weights/model-9.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))

        # prediction
        model.eval()
        batch_size = 10  # 每次预测时将多少张图片打包成一个batch
        with torch.no_grad():
            for ids in range(0, len(img_path_list) // batch_size):
                img_list = []
                for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                    assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                    img = Image.open(img_path).convert('RGB')
                    img = data_transform(img)
                    img_list.append(img)

                # batch img
                # 将img_list列表中的所有图像打包成一个batch
                batch_img = torch.stack(img_list, dim=0)
                # predict class
                output = model(batch_img.to(device)).cpu()
                predict = torch.softmax(output, dim=1)
                probs, classes = torch.max(predict, dim=1)

                for idx, (pro, cla) in enumerate(zip(probs, classes)):
                    print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                     class_indict[str(cla.numpy())],
                                                                     pro.numpy()))

        if i==0:
            c=0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                # print(class_indict[str(cla.numpy())])
                if class_indict[str(cla.numpy())]=='BQ':
                    c+=1
            print(c)
            T+=c

        elif i==1:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'DJ':
                    c += 1
            print(c)
            T += c

        elif i==2:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'DJM':
                    c += 1
            print(c)
            T += c

        elif i==3:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'HHR':
                    c += 1
            print(c)
            T += c

        elif i==4:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'HYF':
                    c += 1
            print(c)
            T += c

        elif i==5:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'JLC':
                    c += 1
            print(c)
            T += c

        elif i==6:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'LD':
                    c += 1
            print(c)
            T += c

        elif i==7:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'LLL':
                    c += 1
            print(c)
            T += c

        elif i==8:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'LQY':
                    c += 1
            print(c)
            T += c

        elif i==9:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'LXY':
                    c += 1
            print(c)
            T += c

        elif i==10:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'LY':
                    c += 1
            print(c)
            T += c

        elif i==11:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'LZG':
                    c += 1
            print(c)
            T += c

        elif i==12:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'MH':
                    c += 1
            print(c)
            T += c

        elif i==13:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'M':
                    c += 1
            print(c)
            T += c

        elif i==14:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'QZH':
                    c += 1
            print(c)
            T += c

        elif i==15:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'TZL':
                    c += 1
            print(c)
            T += c

        elif i==16:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'WJM':
                    c += 1
            print(c)
            T += c

        elif i==17:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'WJY':
                    c += 1
            print(c)
            T += c

        elif i==18:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'YXN':
                    c += 1
            print(c)
            T += c

        elif i==19:
            c = 0
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if class_indict[str(cla.numpy())] == 'ZM':
                    c += 1
            print(c)
            T += c


        print("-----------------------------------------")
    print(T/2)

if __name__ == '__main__':
    main()

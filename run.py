import os
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from fpf_model import net_exam

# dir of dataset by diff algorithm and bpp
dataset_list = {'wow01': "/WOW01/",
                'wow02': "/WOW02/",
                'wow03': "/WOW03/",
                'wow04': "/WOW04/",
                'wow05': "/WOW05/",
                'su01': "/S_UNIWARD01/",
                'su02': "/S_UNIWARD02/",
                'su03': "/S_UNIWARD03/",
                'su04': "/S_UNIWARD04/",
                'su05': "/S_UNIWARD05/",
                'hb01': "/HB01/",
                'hb02': "/HB02/",
                'hb03': "/HB03/",
                'hb04': "/HB04/",
                'hb05': "/HB05/"
                }


# boss dataset customer
class BossDataset(Dataset):
    def __init__(self, dataset_path, stego_type, image_list, dataset_type, transform=None):
        self.transform = transform
        # dir of cover and stego
        ###################################################################################
        self.cover_dir = dataset_path + "/BOSSBASE/cover/"
        self.stego_dir = dataset_path + "/BOSSBASE/stego" + dataset_list[stego_type]
        ######################################################################################

        self.dataset_type = dataset_type
        self.image_list = np.load(image_list)

        # split dataset
        if dataset_type == 0:
            self.image_list = self.image_list[:4000]
        if dataset_type == 1:
            self.image_list = self.image_list[4000:5000]
        if dataset_type == 2:
            self.image_list = self.image_list[5000:]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = os.path.join(self.cover_dir, self.image_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.image_list[file_index])
        # load cover stego
        cover = Image.open(cover_path)
        cover_data = np.array(cover, dtype='float32')

        stego = Image.open(stego_path)
        stego_data = np.array(stego, dtype='float32')

        # match cover and stego
        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        if self.transform:
            data = self.transform(data)
        data = torch.from_numpy(data[:, None, :, :])  # 2 1 H W
        samples = {'images': data, 'labels': torch.from_numpy(label).long()}
        return samples


# train arg
class Argument(object):
    def __init__(self):
        # data_loader
        self.batch_size = 16 // 2
        self.vali_batch_size = 8

        # dataset dir change as yours
        #########################################
        self.dataset_dir = "../0dataset"
        #########################################
        self.transforms = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        # train
        self.epochs = 150
        self.seed = 7
        self.explain = None
        self.net = None

        # optimizer
        self.optimizer = None
        self.use_decay = False
        self.scheduler = None


class SteganalysisNet(object):
    def __init__(self, net_name, criterion=CrossEntropyLoss()):
        if torch.cuda.is_available():
            self.net = net_name.cuda()
            self.criterion = criterion.cuda()
        else:
            self.net = net_name
            self.criterion = criterion
        # save_state
        self.train_result = []
        self.valid_result = []
        self.test_result = []

        self.valid_bestacc = 0
        self.net_state = None

    def train(self, train_loader, optimizer):
        self.net.train()
        epoch_loss = 0.
        epoch_accuracy = 0.
        y_ture = []
        y_pred = []
        for data in train_loader:
            images, labels = data['images'], data['labels']
            images, labels = self.change_shape(images, labels)

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = self.net(images)
            loss = self.criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            y_ture.extend(list(labels.cpu().numpy()))
            y_pred.extend(list(self.out2lable(outputs)))
        # train result
        epoch_accuracy = accuracy_score(y_ture, y_pred)
        epoch_loss /= len(train_loader)
        return epoch_loss.item(), epoch_accuracy

    def validation(self, validation_loader):
        return self.test(validation_loader)

    def test(self, test_loader, ModelState=None):
        if ModelState:
            self.net.load_state_dict(ModelState)

        self.net.eval()
        eval_loss = 0.
        y_ture = []
        y_pred = []
        for data in test_loader:
            images, labels = data['images'], data['labels']
            images, labels = self.change_shape(images, labels)

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                outputs = self.net(images)

            eval_loss += self.criterion(outputs, labels)

            y_ture.extend(list(labels.cpu().numpy()))
            y_pred.extend(list(self.out2lable(outputs)))
        # eval result
        eval_accuracy = accuracy_score(y_ture, y_pred)
        eval_loss /= len(test_loader)
        return eval_loss.item(), eval_accuracy

    def save_result(self, args):
        filename = args.explain
        exam_result = {
            'ValidResult': np.array(self.valid_result),
            'TrainResult': np.array(self.train_result),
            'TestResult': np.array(self.test_result)
        }
        np.save(filename + '.npy', exam_result)
        model_state = {
            'ModelState': self.net_state,
            'bestacc': self.valid_bestacc,
            'args': args.__dict__
        }
        torch.save(model_state, filename + "_m.pth")

    @staticmethod
    def out2lable(outputs):
        _, argmax = torch.max(outputs, 1)
        return argmax.cpu().numpy()

    @staticmethod
    def change_shape(data, label):
        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)
        return data, label


def exam(args):
    print(args.__dict__)
    torch.cuda.manual_seed(args.seed)
    train_dataset = args.train_dataset
    valid_dataset = args.valid_dataset
    test_dataset = args.test_dataset
    print("Generate loaders...")
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.vali_batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.vali_batch_size, shuffle=False, **kwargs)

    print('train_loader have {} iterations, valid_loader have {} iterations'.format(
        len(train_loader), len(valid_loader)))

    print("Generate model")
    analysis_net = SteganalysisNet(args.net)
    _time = time.time()

    optimizer = args.optimizer

    if args.use_decay:
        scheduler = args.scheduler
    for epoch in range(1, args.epochs + 1):
        tloss, taccuracy = analysis_net.train(train_loader, optimizer)
        analysis_net.train_result.append([epoch, tloss, taccuracy])
        Time = time.time() - _time
        if args.use_decay:
            scheduler.step()

        if epoch % 10 == 1:
            print("{:5}\t{:^24}\t{:^24}\t{:<10}".format("", "TRAIN", "VALIDATION", ""))
            print("{:5}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}".format("EPOCH", "tLoss", "tAcc", "vLoss", "vAcc",
                                                                        "time"))
        print("{:5}\t{:<10.6f}\t{:<10.6f}\t{:<10.1f}"
              .format(epoch, tloss, taccuracy, Time))

        if epoch > 100:
            vloss, vaccuracy = analysis_net.validation(valid_loader)
            test_loss, test_accuracy = analysis_net.validation(test_loader)

            analysis_net.valid_result.append([epoch, vloss, vaccuracy])
            analysis_net.test_result.append([epoch, test_loss, test_accuracy])
            print("{:5}\t{:<10.6f}\t{:<10.6f}\t{:<10.6f}\t{:<10.6f}\t{:<10.1f}"
                  .format(epoch, tloss, taccuracy, vloss, vaccuracy, Time))
            print("{:5}\t{:<10}\t{:<10}"
                  .format("EPOCH", "test_loss", "test_acc"))
            print("{:5}\t{:<10.6f}\t{:<10.6f}"
                  .format(epoch, test_loss, test_accuracy))

            if test_accuracy > analysis_net.valid_bestacc:
                analysis_net.valid_bestacc = test_accuracy
                analysis_net.net_state = analysis_net.net.state_dict()
    analysis_net.save_result(args)


def boss(a_type, net, explain):
    args = Argument()
    args.dataset_dir = "../0dataset"
    data_type = a_type
    args.train_dataset = BossDataset(args.dataset_dir, data_type, "list1.npy", 0)
    args.valid_dataset = BossDataset(args.dataset_dir, data_type, "list1.npy", 1)
    args.test_dataset = BossDataset(args.dataset_dir, data_type, "list1.npy", 2)

    args.net = net
    args.explain = 'boss' + data_type + explain

    args.optimizer = SGD(params=args.net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.000)
    args.epochs = 125
    args.batch_size = 8
    args.vali_batch_size = 8

    args.use_decay = True
    DECAY_EPOCH = [100, 150]
    args.scheduler = lr_scheduler.MultiStepLR(args.optimizer, milestones=DECAY_EPOCH, gamma=0.2)
    exam(args)


# train
boss('wow02', net_exam(), '_FPF')
boss('wow04', net_exam(), '_FPF')
boss('su02', net_exam(), '_FPF')
boss('su04', net_exam(), '_FPF')

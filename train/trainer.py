import os
import torch
import yaml
import pandas as pd
from datetime import timedelta
import numpy as np
import matplotlib.pylab as plt
import json
from preprocess import *
from data import *
from optimizers import *
from model import *
from losses import *
from metrics import *


class LstmTrain:
    def __init__(self, cfg, model, data_loader):
        self.cfg = cfg
        self.model = model
        self.project_dir = os.path.dirname(os.path.dirname(__file__))  # 项目路径
        self.data_loader = data_loader
        self.epochs = cfg['Global']['epochs']
        self.save_model_dir = os.path.join(self.project_dir, cfg['Global']['save_model_dir'])
        self.save_image_dir = os.path.join(os.path.dirname(self.save_model_dir), 'images')
        if not os.path.exists(self.save_model_dir): os.makedirs(self.save_model_dir)
        if not os.path.exists(self.save_image_dir): os.makedirs(self.save_image_dir)
        self.loss_fig_flush = cfg['Global']['loss_fig_flush']
        self.print_batch_step = cfg['Global']['print_batch_step']
        self.lr = cfg['Optimizer']['lr']
        self.optimizer = eval(cfg['Optimizer']['name'])(self.model, lr=self.lr)()
        self.loss_fn = eval(cfg['Loss']['name'])()
        self.metric = eval(cfg['Metric']['name'])()
        self.main_indicator = cfg['Metric']['main_indicator']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = cfg['Loss']['name']+'_'+str(cfg['Global']['seq_len'])+'_' + \
                    str(cfg['Global']['output_size'])+'_'+'epochs'+str(self.epochs)

    def run(self):
        total_step = len(self.data_loader)
        epoch_losses = []
        for epoch in range(self.epochs):
            batch_losses = []
            for i, (features, labels) in enumerate(self.data_loader):
                features, labels = features.to(self.device), labels.to(self.device)
                pred = self.model(features)
                loss = self.loss_fn(pred, labels)
                if (i + 1) % self.print_batch_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.
                          format(epoch + 1, self.epochs, i + 1, total_step, loss.item()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses)/len(batch_losses))
            # 绘制loss
            if (epoch + 1) % self.loss_fig_flush == 0:
                self.plot(x_list=[list(range(1, len(epoch_losses)+1))], y_list=[epoch_losses],
                          name='loss.jpg', xlabel='epoch', ylabel='loss', legend=['loss curve'])
        # Save the model checkpoint
        torch.save(self.model.state_dict(), os.path.join(self.save_model_dir, self.name+'.ckpt'))

    def predict(self, features):
        out = self.model(features)
        out = out.cpu().detach().numpy().squeeze(0).tolist()
        return out

    def evaluate(self):
        start_time, end_time = self.cfg['Eval']['start_time'], cfg['Eval']['end_time']
        plt_show = self.cfg['Eval']['plt_show']
        vpt = self.cfg['Eval']['vpt'] if self.cfg['Eval']['vpt'] else 0
        spacing = self.cfg['Global']['spacing']
        seq_len = self.cfg['Global']['seq_len']
        output_size = self.cfg['Global']['output_size']
        # 选择测试区间的真值
        gt_pv = self.data_loader[(self.data_loader['date'] >= pd.to_datetime(start_time))
                                & (self.data_loader['date'] <= pd.to_datetime(end_time))]['pv']
        dates = self.data_loader[(self.data_loader['date'] >= pd.to_datetime(start_time))
                                & (self.data_loader['date'] <= pd.to_datetime(end_time))]['date']
        dates_list = dates.to_list()
        out_list = []
        for i in range(0, len(dates), output_size):
            # 测试数据
            data = self.data_loader[(self.data_loader['date'] >= pd.to_datetime(dates_list[i])-timedelta(minutes=seq_len*spacing))
                                & (self.data_loader['date'] < pd.to_datetime(dates_list[i]))]
            features = data.drop(['date'], axis=1)
            features_numpy = np.expand_dims(features.values, axis=0)
            features_tensor = torch.FloatTensor(features_numpy).to(self.device)
            out = self.model(features_tensor)
            out_list += out.cpu().detach().numpy().squeeze(0).tolist()
        gt_pv_list = gt_pv.values.tolist()
        out_list = out_list[0:len(gt_pv_list)]
        # 逆归一化
        with open(os.path.join(self.project_dir, 'configs', 'min_max.json'), 'r', encoding='utf-8') as fr:
            scaler = fr.read()
            min_max = json.loads(scaler)
        pv_min, pv_max = min_max['pv']['min'], min_max['pv']['max']
        gt_pv_list = [ele*(pv_max-pv_min)+pv_min for ele in gt_pv_list]
        pred_pv_list = [ele*(pv_max-pv_min)+pv_min for ele in out_list]
        if plt_show:
            self.plot([dates_list, dates_list], [gt_pv_list, pred_pv_list], legend=['gt', 'pred'], name='pred_gt.jpg', show=plt_show)
        value = self.metric(gt_pv_list, pred_pv_list)
        # 小于阈值，不计入统计
        vpt_gt_pv_list, vpt_pred_pv_list = [], []
        for i, ele in enumerate(gt_pv_list):
            if ele >= vpt:
                vpt_gt_pv_list.append(ele)
                vpt_pred_pv_list.append(pred_pv_list[i])
        vpt_value = self.metric(vpt_gt_pv_list, vpt_pred_pv_list)
        print(self.main_indicator, round(value, 3), ' ', self.main_indicator+'_vpt', round(vpt_value, 3))

    def plot(self, x_list, y_list, legend=None, name='default.jpg', xlabel=None, ylabel=None, show=False):
        for i, x in enumerate(x_list):
            plt.plot(x, y_list[i], label=legend[i])
        if legend: plt.legend()
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.save_image_dir, self.name+'_'+name))
        if show: plt.show()
        plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = 'configs/lstm.yaml'
    project_dir = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(project_dir, cfg_path), 'r', encoding='UTF-8') as f:
        cfg = yaml.safe_load(f)
    is_train = cfg['Global']['is_train']
    preprocess = eval(cfg['PreProcess']['name'])(cfg)
    # 训练
    data, date = preprocess.load_data()
    data_loader = eval(cfg['Train']['dataset']['name'])(cfg)
    train_loader, test_loader = data_loader(data)
    model = eval(cfg['Architecture']['algorithm'])(cfg).to(device)
    if is_train:
        train = LstmTrain(cfg, model, train_loader)
        train.run()
    else:
        ckpt_path = os.path.join(project_dir, cfg['Global']['ckpt_path'])
        model.load_state_dict(torch.load(ckpt_path))
        train = LstmTrain(cfg, model, pd.concat([data, date], axis=1))
        train.evaluate()










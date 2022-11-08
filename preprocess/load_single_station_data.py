import pandas as pd
import numpy as np
import json, os
import copy


weather = {'中雪': 0, '雪': 1, '大雪': 2, '暴雨': 3,
           '雨': 4, '小雨': 5, '中雨': 6, '大雨': 7, '雨夹雪': 8, '阵雪': 9, '小雪': 10,
           '阴': 11, '霾': 12, '雾': 13, '阵雨': 14, '雷阵雨': 14, '浮尘': 15, '扬沙': 16,
           '多云': 17, '晴': 18}


class LoadSingleStationData:
    def __init__(self, config):
        self.config = config
        self.need_col_name = config['PreProcess']['need_col_name']
        self.pv_excel_path = config['PreProcess']['pv_excel_path']
        self.weather_excel_path = config['PreProcess']['weather_excel_path']
        self.neighbour_point = config['PreProcess']['neighbour_point']

    def load_data(self):
        pv_data = pd.read_excel(self.pv_excel_path).rename(columns={'y': 'pv'})
        weather_data = pd.read_excel(self.weather_excel_path).rename(columns={'sysTime': 'ds'})
        drop_names = []
        # 删除冗余列和天气类型数值化
        for col_name in list(weather_data.columns.values):
            if col_name not in self.need_col_name:
                drop_names.append(col_name)
            if col_name == 'weather':
                weather_data[col_name] = weather_data[col_name].map(self.weather_mapfun)
        weather_data.drop(drop_names, axis=1, inplace=True)
        # pv 和 weather 数据关联
        merge_data = pd.merge(pv_data, weather_data, on='ds', how='inner')
        # 时间特殊处理
        merge_data['date'] = merge_data['ds']
        merge_data['ds'] = merge_data['ds'].map(self.ds_mapfun)  # 采样点
        # 特征归一化
        cols = list(merge_data.columns.values)
        pv_min, pv_max = merge_data['pv'].min(), merge_data['pv'].max()
        without_normal_cols_name = ['date']
        normal_cols_name = [ele for ele in cols if ele not in without_normal_cols_name]
        normal_merge_data, min_max = self.normalization(merge_data, normal_cols_name)
        self.dict2json(min_max)
        if self.neighbour_point:
            for idx in range(self.neighbour_points):
                shift_target = normal_merge_data['pv'].shift(periods=idx + 1)
                normal_merge_data['dist' + '_' + str(idx + 1)] = shift_target
            normal_merge_data.dropna(axis=0, how='all', inplace=True)
        # 将pv列移到最后一列
        columns_name = normal_merge_data.columns.values.tolist()
        columns_name.remove('pv')
        columns_name.append('pv')
        normal_merge_data = normal_merge_data[columns_name]
        # # 如果是训练，删除date
        date = normal_merge_data['date']
        self.config['Global']['normal_max_min'] = [pv_max, pv_min]
        normal_merge_data = normal_merge_data.drop(['date'], axis=1)
        return normal_merge_data, date

    @staticmethod
    def normalization(pd_data, features=[]):
        min_max = dict()
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        for name in features:
            min_max[name] = {'min': float(pd_data[name].min()), 'max': float(pd_data[name].max())}
            pd_data[name] = pd_data[[name]].apply(max_min_scaler)
        return pd_data, min_max

    @staticmethod
    def weather_mapfun(x):
        return weather[x]

    @staticmethod
    def ds_mapfun(x, gap=15):
        scale = 60 / gap
        idx = x.hour * scale + x.minute / gap
        return idx

    def dict2json(self, dict_data):
        p_dir = self.get_project_dir()
        with open(os.path.join(p_dir, 'configs', 'min_max.json'), 'w', encoding="utf-8") as f:
            json.dump(dict_data, f, ensure_ascii=False)

    @staticmethod
    def get_project_dir():
        project_dir = os.path.dirname(os.path.dirname(__file__))
        return project_dir


if __name__ == '__main__':
    pv_excel_path = r'pv.xlsx'
    weather_excel_path = r'weather.xlsx'
    config = None
    data, pv_min_max = LoadSingleStationData(config).load_data()
    print('pv_min_max', pv_min_max)
    # data.values.tolist()
    print(data.tail(50))
    print(data.tail(50).values.tolist())

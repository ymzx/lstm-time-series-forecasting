from model.lstm import LSTM
from model.lstm_seq2seq import Seq2Seq
from model.lstm_seq2seq_fusion import Seq2SeqFusion
from model.cnn_lstm import CNNLSTM


# 定义白名单类名
__all__ = ['LSTM', 'Seq2Seq', 'Seq2SeqFusion', 'CNNLSTM']

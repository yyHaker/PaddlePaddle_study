# -*- coding: utf-8 -*-
import argparse


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")
    # mode (train, evaluate, predict)
    parser.add_argument("--train", default=False, help="train mode")
    parser.add_argument("--evaluate", default=False, help="evaluate mode")
    parser.add_argument("--predict", default=True, help="predict mode")
    # Data
    parser.add_argument("--train_data", default='./data/valid_resource.json', help="train data path")
    parser.add_argument("--valid_data", default='./data/valid_resource.json', help="valid data path")
    # predict mode
    parser.add_argument("--test_data", default='./data/test_resource.json', help='test data path')

    # parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
    # parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
    # parser.add_argument('--test_size', default=10000, type=int, help='Test data size')

    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    # Train
    parser.add_argument('--nof_epoch', default=2000, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # save model path
    parser.add_argument('--save_model_path', default='./models/model', help='save model path')
    # log dir
    parser.add_argument('--log_dir', type=str, default='./logs', help='log dir')
    # GPU
    parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
    # TSP
    parser.add_argument('--nof_points', type=int, default=5, help='Number of points in poem recognition')
    # Network
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
    parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')
    params = parser.parse_args()


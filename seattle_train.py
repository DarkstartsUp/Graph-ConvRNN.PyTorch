import os
import torch
import argparse
import numpy as np
import pandas as pd
import utils
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from graph_convrnn import Graph_ConvRNN

# define constants
DATA_ROOT = 'data/Seattle_Loop_Dataset'


def argument_parsing():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--graph-mode', type=str, default='A',
                        help='Adjacency matrix in {`eye`, `ones`, `A`, `5min`, `10min`, `15min`, `20min`, `25min`}.')
    parser.add_argument('--obs-time-steps', type=int, default=4,
                        help='The number of observed time steps per sample.')
    parser.add_argument('--pred-time-steps', type=int, default=4,
                        help='The number of predicted time steps per sample.')
    parser.add_argument('--in-dims', type=int, default=1,
                        help='The number of input dimensions.')
    parser.add_argument('--out-dims', type=int, default=1,
                        help='The number of output dimensions.')
    parser.add_argument('--num-nodes', type=int, default=323,
                        help='The number of output dimensions.')
    parser.add_argument('--batch-first', action='store_true', default=True,
                        help='Batch first or time-length first.')

    # network config
    parser.add_argument('--rnn-mode', type=str, default='lstm',
                        help='Recursive-NN type, in {`lstm`, `gru`}.')
    parser.add_argument('--gnn-mode', type=str, default='gcn',
                        help='Graph-NN type, in {`gcn`, `gat`}.')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='The number of RNN layers.')
    parser.add_argument('--gcn-hidden-dim', type=list, default=[64, 32, 16],
                        help='GCN hidden dimensions.')
    parser.add_argument('--rnn-hidden-dim', type=list, default=[32, 16, 1],
                        help='RNN hidden dimensions.')
    parser.add_argument('--gnn-bias', action='store_true', default=True,
                        help='Set GCN bias or not.')
    parser.add_argument('--gnn-dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gat-nheads', type=int, default=1,
                        help='The number of GAT heads.')
    parser.add_argument('--return-all-layers', action='store_true', default=False,
                        help='Return outputs of all layers.')

    # training config
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr-step-size', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--shuffle-train-data', action='store_true', default=True,
                        help='Shuffle train data before training.')
    parser.add_argument('--shuffle-val-data', action='store_true', default=True,
                        help='Shuffle validation data before training.')
    parser.add_argument('--shuffle-test-data', action='store_true', default=True,
                        help='Shuffle test data before training.')
    args = parser.parse_args()
    return args


def get_data(args):
    assert args.graph_mode in ['A', '5min', '10min', '15min', '20min', '25min', 'eye', 'ones']
    load_data = pd.read_pickle(os.path.join(DATA_ROOT, 'speed_matrix_2015'))
    load_data = load_data.values  # (105120, 323)
    # sample: from 1 sample per 5 min to 1 sample per 15 min
    load_data = np.array([load_data[i] for i in range(load_data.shape[0]) if i % 3 == 0])

    load_data_x = load_data[:-4, :]  # (B, 323)
    load_data_y = load_data[4:, :]
    data_buffer_x = []
    data_buffer_y = []
    for i in range(load_data_x.shape[0] - 4):
        data_buffer_x.append(load_data_x[i: i + 4])
        data_buffer_y.append(load_data_y[i: i + 4])
    data_buffer_x = np.stack(data_buffer_x, axis=0)  # (B, 4, 323)
    data_buffer_y = np.stack(data_buffer_y, axis=0)
    data_buffer_x = data_buffer_x[:, :, :, np.newaxis]  # (B, 4, 323, 1)
    data_buffer_y = data_buffer_y[:, :, :, np.newaxis]
    sample_num = data_buffer_x.shape[0]
    train_x, train_y = data_buffer_x[:int(sample_num * 0.8)], data_buffer_y[:int(sample_num * 0.8)]
    val_x, val_y = data_buffer_x[int(sample_num * 0.8):int(sample_num * 0.9)], \
                   data_buffer_y[int(sample_num * 0.8):int(sample_num * 0.9)]
    test_x, test_y = data_buffer_x[int(sample_num * 0.9):], data_buffer_y[int(sample_num * 0.9):]

    train_obs_mean, train_pred_mean = np.mean(train_x, axis=(0, 1)), np.mean(train_y, axis=(0, 1))
    train_obs_std, train_pred_std = np.std(train_x, axis=(0, 1)), np.std(train_y, axis=(0, 1))
    val_obs_mean, val_pred_mean = np.mean(val_x, axis=(0, 1)), np.mean(val_y, axis=(0, 1))
    val_obs_std, val_pred_std = np.std(val_x, axis=(0, 1)), np.std(val_y, axis=(0, 1))
    test_obs_mean, test_pred_mean = np.mean(test_x, axis=(0, 1)), np.mean(test_y, axis=(0, 1))
    test_obs_std, test_pred_std = np.std(test_x, axis=(0, 1)), np.std(test_y, axis=(0, 1))

    normed_train_obs = (train_x - train_obs_mean) / train_obs_std  # (B, 4, N, 2)
    normed_train_pred = (train_y - train_pred_mean) / train_pred_std
    normed_val_obs = (val_x - val_obs_mean) / val_obs_std
    normed_val_pred = (val_y - val_pred_mean) / val_pred_std
    normed_test_obs = (test_x - test_obs_mean) / test_obs_std
    normed_test_pred = (test_y - test_pred_mean) / test_pred_std

    if args.graph_mode == 'eye':
        am = np.eye(train_x.shape[2])
    elif args.graph_mode == 'ones':
        am = np.ones((train_x.shape[2], train_x.shape[2]))
    elif 'min' in args.graph_mode:
        am = np.load(os.path.join(DATA_ROOT, 'Loop_Seattle_2015_reachability_free_flow_{}.npy'.format(args.graph_mode)))
    else:  # 'A'
        am = np.load(os.path.join(DATA_ROOT, 'Loop_Seattle_2015_A.npy'))
    normed_am = am / (am.sum(axis=0) + 1e-18)

    return (normed_train_obs, normed_train_pred), (normed_val_obs, normed_val_pred), \
           (normed_test_obs, normed_test_pred), normed_am


def train(args, train_and_valid_data):
    writer = SummaryWriter()
    (train_obs, train_pred), (val_obs, val_pred), am = train_and_valid_data

    train_obs, train_pred = torch.from_numpy(train_obs[:, -args.obs_time_steps:, :, :]).float(), \
                            torch.from_numpy(train_pred[:, :args.pred_time_steps, :, :]).float()
    train_loader = DataLoader(TensorDataset(train_obs, train_pred), batch_size=args.batch_size,
                              shuffle=args.shuffle_train_data)

    val_obs, val_pred = torch.from_numpy(val_obs[:, -args.obs_time_steps:, :, :]).float(), \
                        torch.from_numpy(val_pred[:, :args.pred_time_steps, :, :]).float()
    val_loader = DataLoader(TensorDataset(val_obs, val_pred), batch_size=args.batch_size,
                            shuffle=args.shuffle_val_data)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    model = Graph_ConvRNN(
        input_dim=args.in_dims,
        num_layers=args.num_layers,
        gnn_hidden_dim=args.gcn_hidden_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        num_nodes=args.num_nodes,
        rnn_mode=args.rnn_mode,
        gnn_mode=args.gnn_mode,
        gnn_bias=args.gnn_bias,
        gnn_dropout=args.gnn_dropout,
        batch_first=args.batch_first,
        return_all_layers=args.return_all_layers,
        gat_nheads=args.gat_nheads if args.gnn_mode == 'gat' else None
    ).to(device)
    am = torch.from_numpy(am).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    criterion = nn.MSELoss(reduction='mean')

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch, args.epochs) + '\tlr is {:.6f}.'.format(scheduler.get_lr()[0]))
        model.train()
        train_loss_array = []
        mae_buf, mape_buf, rmse_buf = [], [], []
        for i, _data in enumerate(train_loader):
            _train_obs, _train_pred = _data[0].to(device), _data[1].to(device)  # (B, T, N, C)
            optimizer.zero_grad()

            # prediction of multi-step in auto-regressive style
            single_step_out = []
            model_input = _train_obs
            for j in range(args.pred_time_steps):
                layer_outputs, last_states = model(model_input, am)
                pred = last_states[0][0].unsqueeze(dim=1) if args.rnn_mode == 'lstm' else \
                    last_states[0].unsqueeze(dim=1)
                single_step_out.append(pred)
                model_input = torch.cat((model_input[:, 1:, :, :], pred), dim=1)
            full_out = torch.cat(single_step_out, dim=1)
            loss = criterion(full_out, _train_pred)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            mae_buf.append(utils.masked_mae_np(full_out.cpu().detach().numpy(), _train_pred.cpu().detach().numpy(),
                                               mode='dcrnn'))
            mape_buf.append(utils.masked_mape_np(full_out.cpu().detach().numpy(), _train_pred.cpu().detach().numpy()))
            rmse_buf.append(utils.masked_rmse_np(full_out.cpu().detach().numpy(), _train_pred.cpu().detach().numpy()))

        train_loss_cur = np.mean(train_loss_array)
        train_mae_cur, train_mape_cur, train_rmse_cur = np.mean(mae_buf), np.mean(mape_buf), np.mean(rmse_buf)
        print("Train:\tloss: {:.6f}\tmae: {:.6f}\tmape: {:.6f}\trmse: {:.6f}\t".format(
            train_loss_cur, train_mae_cur, train_mape_cur, train_rmse_cur))

        writer.add_scalar("train/loss", train_loss_cur, epoch)
        writer.add_scalar("train/mae", train_mae_cur, epoch)
        writer.add_scalar("train/mape", train_mape_cur, epoch)
        writer.add_scalar("train/rmse", train_rmse_cur, epoch)

        # validation
        scheduler.step()
        model.eval()
        valid_loss_array = []
        mae_buf, mape_buf, rmse_buf = [], [], []
        for _valid_obs, _valid_pred in val_loader:
            _valid_obs, _valid_pred = _valid_obs.to(device), _valid_pred.to(device)
            single_step_out = []
            model_input = _valid_obs
            for j in range(args.pred_time_steps):
                layer_outputs, last_states = model(model_input, am)
                pred = last_states[0][0].unsqueeze(dim=1) if args.rnn_mode == 'lstm' else \
                    last_states[0].unsqueeze(dim=1)
                single_step_out.append(pred)
                model_input = torch.cat((model_input[:, 1:, :, :], pred), dim=1)
            full_out = torch.cat(single_step_out, dim=1)
            loss = criterion(full_out, _valid_pred)
            valid_loss_array.append(loss.item())
            mae_buf.append(utils.masked_mae_np(full_out.cpu().detach().numpy(), _valid_pred.cpu().detach().numpy(),
                                               mode='dcrnn'))
            mape_buf.append(utils.masked_mape_np(full_out.cpu().detach().numpy(), _valid_pred.cpu().detach().numpy()))
            rmse_buf.append(utils.masked_rmse_np(full_out.cpu().detach().numpy(), _valid_pred.cpu().detach().numpy()))

        valid_loss_cur = np.mean(valid_loss_array)
        valid_mae_cur, valid_mape_cur, valid_rmse_cur = np.mean(mae_buf), np.mean(mape_buf), np.mean(rmse_buf)
        print("valid:\tloss: {:.6f}\tmae: {:.6f}\tmape: {:.6f}\trmse: {:.6f}\t".format(
            valid_loss_cur, valid_mae_cur, valid_mape_cur, valid_rmse_cur))

        writer.add_scalar("valid/loss", valid_loss_cur, epoch)
        writer.add_scalar("valid/mae", valid_mae_cur, epoch)
        writer.add_scalar("valid/mape", valid_mape_cur, epoch)
        writer.add_scalar("valid/rmse", valid_rmse_cur, epoch)
    writer.close()
    return model


def evaluation(args, model, test_data):
    (test_obs, test_pred), am = test_data

    test_obs, test_pred = torch.from_numpy(test_obs[:, -args.obs_time_steps:, :, :]).float(), \
                          torch.from_numpy(test_pred[:, :args.pred_time_steps, :, :]).float()
    test_loader = DataLoader(TensorDataset(test_obs, test_pred), batch_size=args.batch_size,
                             shuffle=args.shuffle_test_data)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    am = torch.from_numpy(am).float().to(device)

    model.eval()
    mae_buf, mape_buf, rmse_buf = [], [], []
    for _test_obs, _test_pred in test_loader:
        _test_obs, _test_pred = _test_obs.to(device), _test_pred.to(device)
        single_step_out = []
        model_input = _test_obs
        for j in range(args.pred_time_steps):
            layer_outputs, last_states = model(model_input, am)
            pred = last_states[0][0].unsqueeze(dim=1) if args.rnn_mode == 'lstm' else \
                last_states[0].unsqueeze(dim=1)
            single_step_out.append(pred)
            model_input = torch.cat((model_input[:, 1:, :, :], pred), dim=1)
        full_out = torch.cat(single_step_out, dim=1)
        mae = utils.masked_mae_np(full_out.cpu().detach().numpy(), _test_pred.cpu().detach().numpy(), mode='dcrnn')
        mape = utils.masked_mape_np(full_out.cpu().detach().numpy(), _test_pred.cpu().detach().numpy())
        rmse = utils.masked_rmse_np(full_out.cpu().detach().numpy(), _test_pred.cpu().detach().numpy())
        mae_buf.append(mae)
        mape_buf.append(mape)
        rmse_buf.append(rmse)
    return np.mean(mae_buf), np.mean(mape_buf), np.mean(rmse_buf)


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_seq, valid_seq, test_seq, am = get_data(args)
    model = train(args, [train_seq, valid_seq, am])
    mae, mape, rmse = evaluation(args, model, [test_seq, am])
    print('mae: {}, mape: {}, rmse: {}'.format(mae, mape, rmse))


if __name__ == '__main__':
    args = argument_parsing()
    main(args)

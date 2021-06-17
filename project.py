from argument_parser import parse_arguments
from argparse import Namespace
from model import Network
from data_converter import generate_train_and_test
from data_loader import StockDataloader
from visualizer import plot_loss, plot_predicted_results
from numpy import inf
from random import sample
from typing import Dict, List, Tuple
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
import sys
import os


def train_and_evaluate(model: Network,
                       train_loader: StockDataloader,
                       test_loader: StockDataloader,
                       optimizer: optim,
                       scheduler: optim,
                       loss_fn: nn,
                       args: Namespace,
                       training_device: torch.device) -> None:
    """
    Train the model and test it
    :param model: Self attention model
    :param train_loader: Training data loader
    :param test_loader: Testing data loader
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param loss_fn: Loss function
    :param args: All arguments
    :param training_device: Training device
    :return: None
    """
    # Setup loss container
    train_losses = [0.0 for _ in range(args.epochs)]
    test_losses = [0.0 for _ in range(args.epochs)]

    # Target company for drawing
    symbol_csv = pd.read_csv(f'data/symbols.csv',
                             delimiter=',',
                             usecols=['Symbol'])
    symbols = symbol_csv['Symbol'][sample(range(len(train_loader)), 5)].tolist()
    del symbol_csv

    min_test_loss = inf
    for epoch in range(args.epochs):
        train_predictions = {sym: [] for sym in symbols}
        test_predictions = {sym: [] for sym in symbols}

        # Train
        info_log(f'[{epoch + 1}/{args.epochs}] Start training')
        avg_loss, train_target_company = train(model=model,
                                               data_loader=train_loader,
                                               optimizer=optimizer,
                                               scheduler=scheduler,
                                               loss_fn=loss_fn,
                                               args=args,
                                               epoch=epoch,
                                               predictions=train_predictions,
                                               training_device=training_device)
        train_losses[epoch] = avg_loss

        # Test
        info_log(f'[{epoch + 1}/{args.epochs}] Start testing')
        avg_loss, test_target_company = test(model=model,
                                             data_loader=test_loader,
                                             loss_fn=loss_fn,
                                             args=args,
                                             epoch=epoch,
                                             preidictions=test_predictions,
                                             training_device=training_device)
        test_losses[epoch] = avg_loss

        if avg_loss < min_test_loss:
            min_test_loss = avg_loss
            checkpoint = {'network': model.state_dict()}
            torch.save(checkpoint, f'models/network_{epoch}_{avg_loss:.4f}.pt')

        # Plot
        info_log(f'[{epoch + 1}/{args.epochs}] Plot losses')
        plot_loss(losses=(train_losses, test_losses), epoch=epoch, label=['Train', 'Test'])
        info_log(f'[{epoch + 1}/{args.epochs}] Plot predicted training results')
        plot_predicted_results(train_predictions=train_predictions,
                               test_predictions=test_predictions,
                               seq_len=args.seq_len)


def train(model: Network,
          data_loader: StockDataloader,
          optimizer: optim,
          scheduler: optim,
          loss_fn: nn,
          args: Namespace,
          epoch: int,
          predictions: Dict[str, List[float]],
          training_device: torch.device) -> Tuple[float, Dict[str, List[float]]]:
    """
    Training
    :param model: Self attention model
    :param data_loader: Training data loader
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param loss_fn: Loss function
    :param args: All arguments
    :param epoch: Current epoch
    :param predictions: Dictionary containing prediction results of target company
    :param training_device: Training device
    :return: Average loss and predicted results
    """
    model.train()
    total_loss, num_batch = 0.0, 0
    for symbol, stock_loader in data_loader:
        info_log(f"[{epoch + 1}/{args.epochs}] Start training stock '{symbol}'")
        for batch_idx, batched_data in enumerate(stock_loader):
            # Get data
            sequence, close = batched_data
            sequence = sequence.to(training_device).type(torch.float)
            close = close.to(training_device).type(torch.float).view(-1, 1)

            # Forward and compute loss
            outputs = model.forward(inputs=sequence, symbol=symbol)
            loss = loss_fn(outputs, close)

            # Calculate gradients and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Record outputs
            try:
                outputs = outputs.view(-1).cpu().detach()
                predictions[symbol] += outputs.tolist()
            except KeyError:
                pass

            # Record loss
            total_loss += loss.item()
            num_batch += 1

            if batch_idx % 20 == 0:
                debug_log(f'[{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(stock_loader)}]   Loss: {loss.item()}')

    return total_loss / num_batch, predictions


def test(model: Network,
         data_loader: StockDataloader,
         loss_fn: nn,
         args: Namespace,
         epoch: int,
         predictions: Dict[str, List[float]],
         training_device: torch.device) -> Tuple[float, Dict[str, List[float]]]:
    """
    Testing
    :param model: Self attention model
    :param data_loader: Testing data loader
    :param args: All arguments
    :param epoch: Current epoch
    :param predictions: Dictionary containing prediction results of target company
    :param training_device Training device
    :return: Average loss and predicted results
    """
    model.eval()
    total_loss, num_batch = 0.0, 0
    for symbol, stock_loader in data_loader:
        info_log(f"[{epoch + 1}/{args.epochs}] Start testing stock '{symbol}'")
        for batch_idx, batched_data in enumerate(stock_loader):
            # Get data
            sequence, close = batched_data
            sequence = sequence.to(training_device).type(torch.float)
            close = close.to(training_device).type(torch.float).view(-1, 1)

            # Forward and compute loss
            outputs = model.forward(inputs=sequence, symbol=symbol)
            loss = loss_fn(outputs, close)

            # Record outputs
            try:
                outputs = outputs.view(-1).cpu().detach()
                predictions[symbol] += outputs.tolist()
            except KeyError:
                pass

            # Record loss
            total_loss += loss.item()
            num_batch += 1

    return total_loss / num_batch, predictions


def inference(model: Network,
              data_loader: StockDataloader,
              args: Namespace,
              training_device: torch.device) -> None:
    """
    Inference
    :param model: Self attention model
    :param data_loader: Testing data loader
    :param args: All arguments
    :param training_device Training device
    :return: None
    """
    model.eval()

    for symbol, stock_loader in data_loader:
        for batch_idx, batched_data in enumerate(stock_loader):
            # Get data
            sequence, close = batched_data
            sequence = sequence.to(training_device).type(torch.float)
            close = close.to(training_device).type(torch.float).view(-1, 1)

            # Forward and compute loss
            outputs = model.forward(inputs=sequence, symbol=symbol)
            # TODO: need to convert the outputs back to stock price
            # TODO: plot test figures (only sample some company for drawing)


def info_log(log: str) -> None:
    """
    Print information log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def debug_log(log: str) -> None:
    """
    Print debug log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity > 1:
        print(f'[\033[93mDEBUG\033[00m] {log}')
        sys.stdout.flush()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Get training device
    training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directory containing models
    if not os.path.exists('models/'):
        os.mkdir('models/')
    if not os.path.exists('figures/'):
        os.mkdir('figures/')

    # Parse arguments
    args = parse_arguments()
    global verbosity
    verbosity = args.verbosity
    info_log(f'Number of epochs: {args.epochs}')
    info_log(f'Number of epochs for warmup: {args.warmup}')
    info_log(f'Learning rate: {args.learning_rate}')
    info_log(f'Batch size: {args.batch_size}')
    info_log(f'Sequence length (consecutive days): {args.seq_len}')
    info_log(f'Number of transformer encoder: {args.num_encoders}')
    info_log(f'Dimension of single attention output: {args.attn_dim}')
    info_log(f'Number of heads for multi-attention: {args.num_heads}')
    info_log(f'Dropout rate: {args.dropout_rate}')
    info_log(f'Hidden size between the linear layers in the network: {args.hidden_size}')
    info_log(f'Inference only or not: {args.inference_only}')
    info_log(f'Training device: {training_device}')

    # Setup model
    info_log('Setup model ...')
    model = Network(batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    num_encoders=args.num_encoders,
                    attn_dim=args.attn_dim,
                    num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate,
                    hidden_size=args.hidden_size).to(training_device)
    if os.path.exists('models/network.pt'):
        checkpoint = torch.load('models/network.pt')
        model.load_state_dict(checkpoint['network'])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / args.warmup))
    loss_fn = nn.MSELoss().to(training_device)

    # Generate data first
    info_log('Generate training and testing data from archive ...')
    generate_train_and_test(root_dir=args.root_dir)

    # Get stock data loader
    info_log('Get data loaders ...')
    train_dataloader = StockDataloader(mode='train', batch_size=args.batch_size, seq_len=args.seq_len)
    test_dataloader = StockDataloader(mode='test', batch_size=args.batch_size, seq_len=args.seq_len)

    if not args.inference_only:
        # Train and test
        train_and_evaluate(model=model,
                           train_loader=train_dataloader,
                           test_loader=test_dataloader,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           loss_fn=loss_fn,
                           args=args,
                           training_device=training_device)
    else:
        # Inference
        info_log('Start inferring')
        inference(model=model,
                  data_loader=test_dataloader,
                  args=args,
                  training_device=training_device)


if __name__ == '__main__':
    verbosity = None
    main()

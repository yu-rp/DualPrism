import argparse
from dig.auggraph.method.DualPrism.specaug import specaug

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDBB',
                    choices=['IMDBB','PROTEINS', 'MUTAG', "REDDITB", 'IMDBM', 'REDDITM5', 'REDDITM12', 'NCI1'])
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training')
parser.add_argument('--model', type = str, default='GIN', 
                    choices=['GCN', 'GIN'])   
parser.add_argument('--nlayers', type = int, default = 4,
                    help='Number of GNN layers.')  
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type = float, default = 0.2,
                    help='Dropout ratio.') 
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train the classifier.')
parser.add_argument('--alpha', type = float, default = 1.0,
                    help='mixup ratio.') 
parser.add_argument('--ckpt_path', type=str, default='path/to/ckpts/', 
                    help='Location for saving checkpoints')
parser.add_argument('--aug_ratio', type=float, default=0.2)
parser.add_argument('--std_dev', type=float, default=1.0)
parser.add_argument('--aug_freq', type=str, default="high", help='low,high,None')
parser.add_argument('--aug_freq_ratio', type=float, default=0.5, help="how much to be auged")
parser.add_argument('--aug_prob', type=float, default=0.5,help="aug with probability aug_prob")

args = parser.parse_args()
      
runner = specaug('path/to/datasets', args.dataset)

runner.train_test(args.batch_size, args.model, cls_nlayers=args.nlayers, 
                  cls_hidden=args.hidden, cls_dropout=args.dropout, cls_lr=args.lr,
                  cls_epochs=args.epochs, ckpt_path=args.ckpt_path,
                  aug_ratio = args.aug_ratio, std_dev=args.std_dev,   aug_prob=args.aug_prob, 
                  aug_freq=args.aug_freq, aug_freq_ratio=args.aug_freq_ratio
                  )

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
from eval import evaluate
import logging
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', help="Action to perform: train, predict, predict_online")
parser.add_argument('--dataset', default="ptg", help="Dataset to use")
parser.add_argument('--split', default='1', help="Data split to use")
parser.add_argument('--batch_size', default=1, type=int, help="Batch size for training")
parser.add_argument('--exp_id', default='mstcn', type=str, help="Experiment ID for model saving")
parser.add_argument('--num_epochs', default=50, type=int, help="Number of training epochs")
parser.add_argument('--causal', action='store_true', help="Use causal convolutions")
parser.add_argument('--graph', action='store_true', help="Use graph structures")
parser.add_argument('--learnable_graph', action='store_true', help="Use learnable graph structures")
parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate")
parser.add_argument('--progress_lw', default=1.0, type=float, help="Loss weight for progress prediction")
parser.add_argument('--graph_lw', default=0.1, type=float, help="Loss weight for graph prediction")

args = parser.parse_args()

# Model parameters
num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = args.batch_size
lr = args.lr 
num_epochs = args.num_epochs

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# File paths
vid_list_file = f"./data/{args.dataset}/splits/train.split{args.split}.bundle"
vid_list_file_tst = f"./data/{args.dataset}/splits/test.split{args.split}.bundle"
features_path = f"./data/{args.dataset}/features/"
gt_path = f"./data/{args.dataset}/groundTruth/"
progress_path = f"./data/{args.dataset}/progress/"
graph_path = f"./data/{args.dataset}/graph/graph.pkl"
mapping_file = f"./data/{args.dataset}/mapping.txt"
model_dir = f"./models/{args.exp_id}/{args.dataset}/split_{args.split}"
results_dir = f"./results/{args.exp_id}/{args.dataset}/epoch{num_epochs}/split_{args.split}"

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('MSTCN')
current_time = datetime.now()
log_filename = current_time.strftime("%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logging
    filename=os.path.join(model_dir, log_filename),   # Name of the log file
    filemode='w',         # 'w' for overwrite each time; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s'  # Include timestamp, log level, and log message
)
ch = logging.StreamHandler()
logger.addHandler(ch)
logger.info(args)

# Read action mapping file
with open(mapping_file, 'r') as file_ptr:
    actions = file_ptr.read().split('\n')[:-1]
# Create action dictionary
actions_dict = dict()
map_delimiter = '|' if args.dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ' '
feature_transpose = True if args.dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else False
for a in actions:
    actions_dict[a.split(map_delimiter)[1]] = int(a.split(map_delimiter)[0])

num_classes = len(actions_dict)

# Initialize trainer
trainer = Trainer(
    num_stages, num_layers, num_f_maps, features_dim, num_classes, 
    causal=args.causal, logger=logger, progress_lw=args.progress_lw, 
    use_graph=args.graph, graph_lw=args.graph_lw, init_graph_path=graph_path, 
    learnable=args.learnable_graph
)

# Perform the specified action
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, progress_path, sample_rate, feature_transpose)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, feature_transpose, map_delimiter)
    evaluate(args.dataset, results_dir, args.split, args.exp_id, args.num_epochs)
elif args.action == 'predict':
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, feature_transpose, map_delimiter)
    evaluate(args.dataset, results_dir, args.split, args.exp_id, args.num_epochs)
elif args.action == "predict_online":
    trainer.predict_online(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, feature_transpose, map_delimiter)
    evaluate(args.dataset, results_dir, args.split, args.exp_id, args.num_epochs)
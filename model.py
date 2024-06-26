import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import tqdm
import pickle

# Define the MultiStageModel class
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, causal=False, use_graph=True, **graph_args):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, causal)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, causal)) for s in range(num_stages-1)])
        self.use_graph = use_graph
        if use_graph:
            self.graph_learner = TaskGraphLearner(**graph_args)

    def forward(self, x, mask):
        out, out_app = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        outputs_app = out_app.unsqueeze(0)
        for s in self.stages:
            out, out_app = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            outputs_app = torch.cat((outputs_app, out_app.unsqueeze(0)), dim=0)
        return outputs, outputs_app

# Define the ProbabilityProgressFusionModel class
class ProbabilityProgressFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(ProbabilityProgressFusionModel, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv1d(num_classes*2, num_classes, 1)

    def forward(self, in_cls, in_prg):
        ### in_cls: batch_size x num_classes x T
        ### in_prg: batch_size x num_classes x T
        # Concatenate classification and progress inputs
        input_concat = torch.cat((in_cls, in_prg), dim=1)
        out = self.conv(input_concat)
        return out

# Define the TaskGraphLearner class
class TaskGraphLearner(nn.Module):
    def __init__(self, init_graph_path, learnable=False, reg_weight=0.01, eta=0.01):
        super(TaskGraphLearner, self).__init__()
        with open(init_graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        matrix_pre, matrix_suc = self.graph['matrix_pre'], self.graph['matrix_suc']
        self.matrix_pre = nn.Parameter(torch.from_numpy(matrix_pre).float(), requires_grad=learnable)
        self.matrix_suc = nn.Parameter(torch.from_numpy(matrix_suc).float(), requires_grad=learnable)
        self.learnable = learnable
        if learnable:
            self.matrix_pre_original = nn.Parameter(self.matrix_pre, requires_grad=False)
            self.matrix_suc_original = nn.Parameter(self.matrix_suc, requires_grad=False)
        self.reg_weight = reg_weight
        self.eta = eta

    def forward(self, cls, prg):
        action_prob = F.softmax(cls, dim=1)
        prg = torch.clamp(prg, min=0, max=1)
        completion_status, _ = torch.cummax(prg, dim=-1)
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        graph_loss = ((alpha_pre + alpha_suc) * action_prob).mean()
        if self.learnable:
            regularization = torch.mean((self.matrix_pre - self.matrix_pre_original) ** 2)
            return graph_loss + self.reg_weight * regularization
        return graph_loss

    def inference(self, cls, prg):
        action_prob = F.softmax(cls, dim=1)
        prg = torch.clamp(prg, min=0, max=1)
        completion_status, _ = torch.cummax(prg, dim=-1)
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        logits = cls - self.eta * (alpha_pre + alpha_suc)
        return logits

# Define the SingleStageModel class
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal=causal)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        ### Action Progress Prediction (APP) module
        self.gru_app = nn.GRU(num_f_maps, num_f_maps, num_layers=1, batch_first=True, bidirectional=not causal)
        self.conv_app = nn.Conv1d(num_f_maps, num_classes, 1)
        self.prob_fusion = ProbabilityProgressFusionModel(num_classes)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        prob_out = self.conv_out(out) * mask[:, 0:1, :]
        progress_out, _ = self.gru_app(out.permute(0, 2, 1))
        progress_out = progress_out.permute(0, 2, 1)
        progress_out = self.conv_app(progress_out) * mask[:, 0:1, :]
        out = self.prob_fusion(prob_out, progress_out)
        out = out * mask[:, 0:1, :]
        return out, progress_out

# Define the DilatedResidualLayer class
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, filter_size=3, causal=False):
        super(DilatedResidualLayer, self).__init__()
        self.causal = causal
        self.dilation = dilation
        padding = int(dilation * (filter_size-1) / 2)
        if causal:
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, filter_size, padding=padding*2, padding_mode='replicate', dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, filter_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        if self.causal:
            out = out[..., :-self.dilation*2]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

# Define the Trainer class
class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, causal, logger, progress_lw=1,
                 use_graph=True, graph_lw=0.1, init_graph_path='', learnable=True):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes, causal, 
                     use_graph=use_graph, init_graph_path=init_graph_path, learnable=learnable)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.progress_lw = progress_lw
        self.use_graph = use_graph
        self.graph_lw = graph_lw
        self.logger = logger

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_progress_loss = 0
            epoch_graph_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, batch_progress_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_progress_target, mask = batch_input.to(device), batch_target.to(device), batch_progress_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, progress_predictions = self.model(batch_input, mask)

                loss = 0
                progress_loss = 0
                for p, progress_p in zip(predictions, progress_predictions):
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :, 1:])
                    progress_loss += self.mse(progress_p, batch_progress_target).mean()

                loss += self.progress_lw * progress_loss
                epoch_progress_loss += self.progress_lw * progress_loss.item()

                graph_loss = self.model.graph_learner(predictions[-1], progress_predictions[-1])
                loss += self.graph_lw * graph_loss
                epoch_graph_loss += self.graph_lw * graph_loss.item()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            self.logger.info("[epoch %d]: epoch loss = %f, progress loss = %f, graph loss = %f, acc = %f" % (epoch + 1, 
                                                              epoch_loss / len(batch_gen.list_of_examples),
                                                              epoch_progress_loss / len(batch_gen.list_of_examples),
                                                              epoch_graph_loss / len(batch_gen.list_of_examples),
                                                              float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, feature_transpose=False, map_delimiter=' '):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, progress_predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])
                _, predicted = torch.max(final_predictions.data, 1)

                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(map_delimiter.join(recognition))
                f_ptr.close()

    def predict_online(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, feature_transpose=False, map_delimiter=' '):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                n_frames = input_x.shape[-1]
                recognition = []
                for frame_i in tqdm.tqdm(range(n_frames)):
                    curr_input_x = input_x[:, :, :frame_i+1]
                    predictions, progress_predictions = self.model(curr_input_x, torch.ones(curr_input_x.size(), device=device))
                    final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])
                    _, predicted = torch.max(final_predictions.data, 1)
                    predicted = predicted.squeeze(0)
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[-1].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(map_delimiter.join(recognition))
                f_ptr.close()

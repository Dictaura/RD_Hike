import torch
from torch.utils import data

from e2efold.e2efold.models import  ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.e2efold.models import Lag_PP_NN, RNA_SS_e2e, Lag_PP_zero, Lag_PP_perturb
from e2efold.e2efold.models import Lag_PP_mixed, ContactAttention_simple
from e2efold.e2efold.common.utils import *
from e2efold.e2efold.common.config import process_config
from e2efold.e2efold.evaluation import all_test_only_e2e

##############################################################
# function
##############################################################

def decode_contact(contact, seq_len):
    """
    decode the adjacent matrix
    :param contact: adjacent matrix
    :param seq_len: sequence length
    :return: contact dictionary and dot-bracket
    """
    contact = contact[:seq_len, :seq_len]
    structure = np.where(contact)
    pair_dict = dict()
    for i in range(seq_len):
        pair_dict[i] = -1
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]
    dotB = ""
    for i in range(seq_len):
        if (pair_dict[i] == -1):
            dotB += "."
        elif (i < pair_dict[i]):
            dotB += "("
        else:
            dotB +=")"

    return pair_dict, dotB



##############################################################
# test
##############################################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# seed_torch(0)
#
# seq_len = 600
#
# args = get_args()
#
# config_file = args.config

############################################################
# config
############################################################
# config = process_config(config_file)
#
# os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
#
# d = config.u_net_d
# BATCH_SIZE = config.BATCH_SIZE
# OUT_STEP = config.OUT_STEP
# LOAD_MODEL = config.LOAD_MODEL
# pp_steps = config.pp_steps
# pp_loss = config.pp_loss
# data_type = config.data_type
# model_type = config.model_type
# pp_type = '{}_s{}'.format(config.pp_model, pp_steps)
# rho_per_position = config.rho_per_position
# e2e_model_path = 'e2efold/models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(model_type,
#     pp_type,d, data_type, pp_loss,rho_per_position)
# epoches_third = config.epoches_third
# evaluate_epi = config.evaluate_epi
# step_gamma = config.step_gamma
# k = config.k
#
# # if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # seed everything for reproduction
# seed_torch(0)
#
# seq_len = 600
#
# if model_type =='test_lc':
#     contact_net = ContactNetwork_test(d=d, L=seq_len).to(device)
# if model_type == 'att6':
#     contact_net = ContactAttention(d=d, L=seq_len).to(device)
# if model_type == 'att_simple':
#     contact_net = ContactAttention_simple(d=d, L=seq_len).to(device)
# if model_type == 'att_simple_fix':
#     contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len,
#         device=device).to(device)
# if model_type == 'fc':
#     contact_net = ContactNetwork_fc(d=d, L=seq_len).to(device)
# if model_type == 'conv2d_fc':
#     contact_net = ContactNetwork(d=d, L=seq_len).to(device)
#
# # need to write the class for the computational graph of lang pp
# if pp_type=='nn':
#     lag_pp_net = Lag_PP_NN(pp_steps, k).to(device)
# if 'zero' in pp_type:
#     lag_pp_net = Lag_PP_zero(pp_steps, k).to(device)
# if 'perturb' in pp_type:
#     lag_pp_net = Lag_PP_perturb(pp_steps, k).to(device)
# if 'mixed'in pp_type:
#     lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position).to(device)
#
# rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)
#
# if LOAD_MODEL and os.path.isfile(e2e_model_path):
#     print('Loading e2e model...')
#     rna_ss_e2e.load_state_dict(torch.load(e2e_model_path))
#
# contact_net.eval()
# lag_pp_net.eval()
#
# ct_list = list()
#
# seqs = ['UGGUGGCUAUAGCAAAAAUGAACCACCCGAUCUCAUCUCGAACUCGGAAGUGAAACUUUUUAGCGCUGAUGGUACUUGAAAAGGGAGAGUAGGUCGCCGCCAAGUU']
#
# seq_embeddings =  list(map(seq_encoding, seqs))
# seq_embeddings = list(map(lambda x: padding(x, seq_len),
#     seq_embeddings))
# seq_embeddings = np.array(seq_embeddings)
# seq_lens = torch.Tensor(np.array(list(map(len, seqs)))).int()
#
# seq_embedding_batch = torch.Tensor(seq_embeddings).float().to(device)
#
# state_pad = torch.zeros(1,2,2).to(device)
#
# PE_batch = get_pe(seq_lens, seq_len).float().to(device)
# with torch.no_grad():
#     pred_contacts = contact_net(PE_batch,
#         seq_embedding_batch, state_pad)
#     a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)
#
# final_pred = (a_pred_list[-1].cpu()>0.5).float()
#
# for i in range(final_pred.shape[0]):
#     # ct_tmp = contact2ct(final_pred[i].cpu().numpy(),
#     #     seq_embeddings[i], seq_lens.numpy()[i])
#     # ct_list.append(ct_tmp)
#     ct_tmp = decode_contact(final_pred[i].cpu().numpy(),
#                             seq_lens.numpy()[i])
#
# print(ct_list)

########################################################
# class
########################################################

class Str_Predictor:
    def __init__(self, device=None, max_len=600):
        args = get_args()
        config_file = args.config
        config = process_config(config_file)
        if (device != None):
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_torch(0)
        self.seq_len = max_len
        self.args = get_args()
        self.config_file = args.config
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

        self.d = config.u_net_d
        self.BATCH_SIZE = config.BATCH_SIZE
        self.OUT_STEP = config.OUT_STEP
        self.LOAD_MODEL = config.LOAD_MODEL
        self.pp_steps = config.pp_steps
        self.pp_loss = config.pp_loss
        self.data_type = config.data_type
        self.model_type = config.model_type
        self.pp_type = '{}_s{}'.format(config.pp_model, self.pp_steps)
        self.rho_per_position = config.rho_per_position
        self.e2e_model_path = 'e2efold/models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(self.model_type,
                                                                                    self.pp_type, self.d, self.data_type, self.pp_loss,
                                                                                    self.rho_per_position)
        self.epoches_third = config.epoches_third
        self.evaluate_epi = config.evaluate_epi
        self.step_gamma = config.step_gamma
        self.k = config.k

        # model type to contact net
        if self.model_type == 'test_lc':
            self.contact_net = ContactNetwork_test(d=self.d, L=self.seq_len).to(self.device)
        if self.model_type == 'att6':
            self.contact_net = ContactAttention(d=self.d, L=self.seq_len).to(self.device)
        if self.model_type == 'att_simple':
            self.contact_net = ContactAttention_simple(d=self.d, L=self.seq_len).to(self.device)
        if self.model_type == 'att_simple_fix':
            self.contact_net = ContactAttention_simple_fix_PE(d=self.d, L=self.seq_len,
                                                         device=self.device).to(self.device)
        if self.model_type == 'fc':
            self.contact_net = ContactNetwork_fc(d=self.d, L=self.seq_len).to(self.device)
        if self.model_type == 'conv2d_fc':
            self.contact_net = ContactNetwork(d=self.d, L=self.seq_len).to(self.device)

        if self.pp_type == 'nn':
            self.lag_pp_net = Lag_PP_NN(self.pp_steps, self.k).to(self.device)
        if 'zero' in self.pp_type:
            self.lag_pp_net = Lag_PP_zero(self.pp_steps, self.k).to(self.device)
        if 'perturb' in self.pp_type:
            self.lag_pp_net = Lag_PP_perturb(self.pp_steps, self.k).to(self.device)
        if 'mixed' in self.pp_type:
            self.lag_pp_net = Lag_PP_mixed(self.pp_steps, self.k, self.rho_per_position).to(self.device)

        self.rna_ss_e2e = RNA_SS_e2e(self.contact_net, self.lag_pp_net)

        if self.LOAD_MODEL and os.path.isfile(self.e2e_model_path):
            print('Loading e2e model...')
            self.rna_ss_e2e.load_state_dict(torch.load(self.e2e_model_path))

        self.contact_net.eval()
        self.lag_pp_net.eval()

    def predict(self, seqs):
        seq_embeddings = list(map(seq_encoding, seqs))
        seq_embeddings = list(map(lambda x: padding(x, self.seq_len),
                                  seq_embeddings))
        seq_embeddings = np.array(seq_embeddings)
        seq_lens = torch.Tensor(np.array(list(map(len, seqs)))).int()

        seq_embedding_batch = torch.Tensor(seq_embeddings).float().to(self.device)

        state_pad = torch.zeros(1, 2, 2).to(self.device)

        PE_batch = get_pe(seq_lens, self.seq_len).float().to(self.device)
        with torch.no_grad():
            pred_contacts = self.contact_net(PE_batch,
                                        seq_embedding_batch, state_pad)
            a_pred_list = self.lag_pp_net(pred_contacts, seq_embedding_batch)

        final_pred = (a_pred_list[-1].cpu() > 0.5).float()

        ct_list = list()

        for i in range(final_pred.shape[0]):
            ct_tmp = decode_contact(final_pred[i].cpu().numpy(),
                                    seq_lens.numpy()[i])
            ct_list.append(ct_tmp)

        return ct_list

    def predict_base_list(self, seqs_list):
        """
        predict structures from sequences of base_list
        :param seqs_list: sequences of base_list
        :return: list of dict and dot-bracket
        """
        seq_embeddings = seqs_list
        seq_embeddings = list(map(lambda x: padding(x, self.seq_len),
                                  seq_embeddings))
        seq_embeddings = np.array(seq_embeddings)
        seq_lens = torch.Tensor(np.array(list(map(len, seqs_list)))).int()

        seq_embedding_batch = torch.Tensor(seq_embeddings).float().to(self.device)

        state_pad = torch.zeros(1, 2, 2).to(self.device)

        PE_batch = get_pe(seq_lens, self.seq_len).float().to(self.device)
        with torch.no_grad():
            pred_contacts = self.contact_net(PE_batch,
                                             seq_embedding_batch, state_pad)
            a_pred_list = self.lag_pp_net(pred_contacts, seq_embedding_batch)

        final_pred = (a_pred_list[-1].cpu() > 0.5).float()

        ct_list = list()

        for i in range(final_pred.shape[0]):
            ct_tmp = decode_contact(final_pred[i].cpu().numpy(),
                                    seq_lens.numpy()[i])
            ct_list.append(ct_tmp)

        return ct_list
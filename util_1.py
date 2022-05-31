## EXPERIMENTAL
## OUTDATED
## Tried to use a different loss method to compensate for the divergence of the predicted probabilities of ASM
## Will not add too many comments
# ## To separate into variables and labels
import torch
from torch.autograd import Variable
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
src_mask_on = False
num_cols_to_remove = 1  ## Psuedo_outcome - this is removed as it is the lable
debug = False
## Tunable parameters
# all_accuracy = True  ## Uses all accuracies or just test_idx : test_end_idx
# drop_out_prob = 0.1
# normalize_learning_rate = True  # Normalisation of LR for imbalanced classes
# aggregate_loss = True  # Mini batch losses for back prop
## Making my own mini batch loss
# mini_batch_loss = 20
# first_regimen_only = True  # First regimen just in case data-pre-processing did not delete it
# use_weighted_acc = True  ## For model training - uneven split of succesful and unsuccessful outcomes
# batch_size = 1  ## Because we have a small data set, keep it as 1
## Experimental parameters (outdated) - Not maintained
## This means as the project went on, and these variables went to TRUE or FALSE, they were not maintained to make sure the code still worked
# use_regimen_in_decoder = False  # Not working yet (not working yet for True)
# use_pe = True  # Might be worth deleting the positional encoder (not working yet for False)
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# class Norm(nn.Module):
#     def __init__(self, d_model, eps=1e-6):
#         super().__init__()
#
#         self.size = d_model
#         # create two learnable parameters to calibrate normalisation
#         self.alpha = nn.Parameter(torch.ones(self.size))
#         self.bias = nn.Parameter(torch.zeros(self.size))
#         self.eps = eps
#
#     def forward(self, x):
#         x_norm = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + self.eps)
#         norm = self.alpha * x_norm + self.bias
#         return norm

# # ### Transformer model
# class PositionalEncoder(nn.Module):
#     def __init__(self, d_model, max_seq_len=20, dropout_rate=0):
#         super().__init__()
#         self.d_model = d_model
#
#         pe = torch.zeros(max_seq_len, d_model)
#         for pos in range(max_seq_len):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
#                 pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x, grad_req=False):
#         # make embeddings relatively larger
#         x = x * math.sqrt(self.d_model)
#         # add constant to embedding
#         seq_len = x.shape[1]##########batch_size大小
#         pe = Variable(self.pe[:, :seq_len], requires_grad=grad_req)
#         if x.is_cuda:
#             pe.cuda()
#
#         if use_pe:
#             x = x + pe
#         #         return self.dropout(x)
#         return x
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout_rate=0.):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear_final = nn.Linear(d_model, d_model)
        self.attention_inter = attention_inter()
        self.layer_norm = nn.BatchNorm1d(d_model)
            # LayerNorm(d_model, eps=1e-12)
        #
    def forward(self, k, v, q,  mask=None):
        bs = q.size(0)
        residual = q
        # perform linear operation and split into N heads

        # k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        # q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        # v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        # # transpose to get dimensions bs * N * sl * d_model
        # k = k.transpose(1, 2)
        # q = q.transpose(1, 2)
        # v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores, attn_weights = self.attention_inter(q, k, v)
        # concatenate heads and put through final linear layer
        #         print("scores shape",scores.shape)
        concat = scores
        # concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        #         print("concat shape",concat.shape)
        output = self.linear_final(concat)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm((residual + output).squeeze())
        return output.unsqueeze(1), attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout_rate=0.05):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        # self.bn1 = nn.BatchNorm1d(d_ff)
        self.dropout = nn.Dropout(0.1)
        # self.linear_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.BatchNorm1d(d_model)
        # LayerNorm(d_model, eps=1e-12)
        #
        # LayerNorm(d_model, eps=1e-12)
        # self.bn2 = nn.BatchNorm1d(d_model)
    def forward(self, x):
        out = self.dropout(F.relu(self.linear_1(x)))
        # out = self.dropout(self.linear_2(out))
        out = self.layer_norm((x + out).squeeze())
        return out.unsqueeze(1)

class attention_inter(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_dropout = nn.Dropout(0.07)
    def forward(self, q, k, v):
        q = q.unsqueeze(-1)  ###64*2*29*1
        k = k.unsqueeze(-1)  ###64*2*29*1
        v = v.unsqueeze(-1)  ###64*2*29*1
        scores = torch.matmul(q, k.transpose(-2, -1))  ##64*2*29*29
        scores = F.softmax(scores, dim=-1)
        attn_weights_inter = copy.copy(scores)
        # if dropout is not None:
        scores = self.attn_dropout(scores)
        # 添加dropout
        # attention = self.dropout(scores)
        output = torch.matmul(scores, v)
        if debug:
            print("output shape", output.shape)
        output = output.squeeze(-1)
        return output, attn_weights_inter
# build an encoder layer with one multi-head attention layer and one # feed-forward layerclass EncoderLayer(nn.Module):
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
    def forward(self, x):
        y, attn_weights = self.attn(x, x, x)
        output = self.ff(y)
        return output, attn_weights
# # build a decoder layer with two multi-head attention layers and
# # one feed-forward layerclass DecoderLayer(nn.Module):
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()

        # self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)  # .cuda()

    def forward(self, e_outputs):
        # dec_output, self_attention = self.attn_1(dec_inputs, dec_inputs, dec_inputs)
        dec_output, context_attention = self.attn_2(e_outputs, e_outputs, e_outputs)
        dec_output = self.ff(dec_output)
        return dec_output, context_attention

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(vocab_size, d_model)
            # Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, d_ff), N)
        self.layer_norm = nn.BatchNorm1d(d_model)
        # LayerNorm(d_model, eps=1e-12)
        # nn.BatchNorm1d(d_model)
            # LayerNorm(d_model, eps=1e-12)
        # self.ac_layer = nn.ELU(alpha=1.0, inplace=False)

    def forward(self, src):
        output = self.embed(src)
        output = self.layer_norm(torch.squeeze(output))
        output = output.unsqueeze(1)
        # output = self.pe(output, grad_req)
        attn = []
        for i in range(self.N):
            output, attention = self.layers[i](output)
            attn.append(attention)
        return output, attn

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, d_ff):
        super().__init__()
        self.N = N
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, d_ff), N)
        self.layer_norm = nn.BatchNorm1d(d_model)
        # LayerNorm(d_model, eps=1e-12)

    def forward(self, trg, e_outputs, grad_req=False):

        # x = trg.long()

        # if debug:
        #     print("x shape before", x.shape)
        # output = self.pe(x, grad_req)
        # output = Variable(torch.zeros(64,1,300), requires_grad=grad_req).cuda()

        # if debug:
        #     print("x shape after", x.shape)
        # self_attentions = []
        context_attentions = []
        for i in range(self.N):
            output, context_attn = self.layers[i](e_outputs)
            # self_attentions.append(self_attn)
            context_attentions.append(context_attn)
        return output, context_attentions

class Transformer(nn.Module):
    def __init__(self, src_var, trg_label, d_model, N, heads, d_ff):
        super().__init__()
        self.trg_label = trg_label
        self.encoder = Encoder(src_var, d_model, N, heads, d_ff)
        self.decoder = Decoder(d_model, N, heads, d_ff)
        self.classifier = nn.Linear(d_model, 1)
        self.ac = nn.Sigmoid()
        self.attention_inter = attention_inter()
    def forward(self, src, trg, grad_req=False):
        src, inter_atten = self.attention_inter(src,src,src)
        if debug:
            print("after encoder")
        e_output, enc_attn = self.encoder(src)

        # if debug:
        #     print("after decoder")
        d_output, ctx_attn= self.decoder(trg, e_output, grad_req)
        output = self.ac(self.classifier(d_output))
            # output = F.softmax(self.classifier(d_output),dim=2)
        return output,d_output # we don't perform softmax on the output as this will be handled

# def create_masks(src, trg):
#     if (src_mask_on):
#         src_mask = torch.zeros(1, src.shape[1], src.shape[1])
#     else:
#         src_mask = torch.ones(1, src.shape[1], src.shape[1])
#     trg_mask = torch.zeros(1, trg.shape[1], trg.shape[1])
#
#     ctr = 1
#     if (src_mask_on):
#         for i in range(src.shape[1]):
#             for k in range(ctr):
#                 src_mask[0][i][k] = True
#             ctr = ctr + 1
#
#     ctr = 1
#     for i in range(trg.shape[1]):
#         for k in range(ctr):
#             trg_mask[0][i][k] = True
#         ctr = ctr + 1
#
#     return src_mask, trg_mask

from utils import LinearAct
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from param import args

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


class GeneratorExtractor(nn.Module):
    def __init__(self, ntoken, ctx_size):
        super().__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim


        self.img_fc = nn.Sequential(
            nn.LayerNorm(ctx_size),
            nn.Dropout(0.3),
            nn.Linear(ctx_size, hidden_size),
            nn.Tanh(),
        )

        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        self.lstm_hist = nn.LSTM(self.emb_dim, int(hidden_size/2), batch_first=True, bidirectional=True)
        self.lstm_hist_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        self.lstm_concept = nn.LSTM(self.emb_dim, int(hidden_size/2), batch_first=True, bidirectional=True)

        self.lang_int = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(hidden_size)
        )

        self.lang_int_img = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(hidden_size)
        )
        self.projection = LinearAct(hidden_size, ntoken)
        self.gate = LinearAct(hidden_size, 3)

        self.box_encode = LinearAct(6, hidden_size)
        self.img_box_encode = LinearAct(hidden_size*2, hidden_size, 'tanh')


        self.pe = torch.zeros(6, hidden_size)  
        position = torch.arange(0, 6).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * - (math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
    def forward(self, imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len, h0, c0, h1, c1, h2, c2,ctx_mask=None):


        bsz, num_img, c_num, c_len = concept.size()
        hist_len = hists.size(1)
        
        with torch.no_grad():
            h_1 = torch.zeros(2, bsz*hist_len, int(self.hidden_size/2)).cuda()
            c_1 = torch.zeros(2, bsz*hist_len, int(self.hidden_size/2)).cuda()

            h_2 = torch.zeros(2, bsz*num_img*c_num, int(self.hidden_size/2)).cuda()
            c_2 = torch.zeros(2, bsz*num_img*c_num, int(self.hidden_size/2)).cuda()

            h_3 = torch.zeros(1, bsz, self.hidden_size).cuda()
            c_3 = torch.zeros(1, bsz, self.hidden_size).cuda()

            pe = torch.zeros(bsz, 6, self.hidden_size).cuda()


        imgs = self.img_fc(imgs)
        boxes = self.box_encode(boxes)

        for i in range(bsz):
            for j in range(img_leng[i]):
                pe[i, j, :] = self.pe[img_leng[i]-j-1, :]

        

        concept = concept.view(bsz * num_img * c_num, c_len)
        concept_embeds = self.w_emb(concept)
        concept_embeds = self.drop(concept_embeds)

        imgs_mask = imgs_mask.view(bsz, -1)

        concept_enc, (h2, c2) = self.lstm_concept(concept_embeds, (h_2, c_2))
        concept_enc = concept_enc.contiguous().view(bsz, num_img, c_num, c_len, -1)

        concept_enc = concept_enc + pe.unsqueeze(-2).unsqueeze(-2)

        imgs = self.img_box_encode(torch.cat([imgs, boxes], dim=-1))
        imgs = imgs + pe.unsqueeze(-2)

        api_len = api_gt.size(1)
        api_embeds = self.w_emb(api_gt)      
        api_embeds = self.drop(api_embeds)

        api_enc, (h0, c0) = self.lstm(api_embeds, (h0, c0))
        api_enc = self.drop(api_enc)


        bsz, h_len, u_len = hists.size()

        hist_mask = (hists.view(bsz, h_len*u_len) != 0).float()

        hist_len_mask = (hists.view(bsz, h_len, u_len).sum(-1) != 0).float()
       

        hists = hists.view(bsz * h_len, u_len)
        hists_embeds = self.w_emb(hists)
        hists_embeds = self.drop(hists_embeds)

        hists_enc, (h1, c1) = self.lstm_hist(hists_embeds, (h_1, c_1))
        
        hist_attenion_map = hists_enc.contiguous().view(bsz, h_len*u_len, -1)

        hists_enc_collapse = gethidden(hists_enc, hists_len)
        hists_enc_collapse = hists_enc_collapse.view(bsz, h_len, -1)

        hists_enc_collapse, (h3, c3) = self.lstm_hist_2(hists_enc_collapse, (h_3, c_3))

        imgs = imgs.view(bsz, 6*100, -1)
        
        imgs_hist = self.attention_img(imgs, hists_enc_collapse, imgs_mask, hist_len_mask)
        x_i = self.attention(api_enc, imgs_hist, imgs_mask)
        

        concept_mask = (concept.view(bsz, num_img*c_num*c_len) != 0).float()
        concept_enc_map = concept_enc.view(bsz, num_img*c_num*c_len, -1)
        
        concept_enc_map = self.attention(concept_enc_map, imgs_hist, imgs_mask)

        concept_map = self.attention_map(x_i, concept_enc_map, concept_mask)
        hist_map = self.attention_map(api_enc, hist_attenion_map, hist_mask)

        logit = self.projection(x_i)

        logit_sm = torch.softmax(logit, dim=-1)
        hist_ext = hists.view(bsz, 1, h_len*u_len).repeat(1,api_len, 1)
        concept_ext = concept.view(bsz, 1, num_img*c_num*c_len).repeat(1,api_len, 1)

        gated_x = self.gate(x_i)
        gate = torch.softmax(gated_x, dim=-1)

        logit_sm = logit_sm * gate[:, :, 0:1]
        hist_map = hist_map * gate[:, :, 1:2]
        concept_map = concept_map * gate[:, :, 2:3]
        logit_new = logit_sm.scatter_add(2, hist_ext, hist_map)
        logit_new = logit_new.scatter_add(2, concept_ext, concept_map)

        return logit_new, h0, c0, h1, c1, h2, c2, gate




    def attention_img(self, insts_enc, visual, mask1, mask2):

        sim = torch.matmul(insts_enc, visual.transpose(-2,-1))

        if mask1 is not None:
            sim = mask_logits(sim, mask2.unsqueeze(1))   

        sim_l = torch.softmax(sim, dim=-1) 

        ltv = torch.matmul(sim_l, visual) 

        inst_new = self.lang_int_img(torch.cat([insts_enc, ltv, insts_enc*ltv], dim=-1))

        return inst_new

    def attention(self, insts_enc, visual, mask):

        sim = torch.matmul(insts_enc, visual.transpose(-2,-1))

        if mask is not None:
            sim = mask_logits(sim, mask.unsqueeze(1))   
        sim_l = torch.softmax(sim, dim=-1)

        ltv = torch.matmul(sim_l, visual) 

        inst_new = self.lang_int(torch.cat([insts_enc, ltv, insts_enc*ltv], dim=-1))

        return inst_new


    def attention_map(self, insts_enc, visual, visual_mask):

        sim = torch.matmul(insts_enc, visual.transpose(-2,-1))

        if visual_mask is not None:
            sim = mask_logits(sim, visual_mask.unsqueeze(1))   
        sim = torch.softmax(sim, dim=-1) 

        return sim

def gethidden(hists_enc, hists_len):
    bsz, _, d = hists_enc.size()
    hists_len = hists_len.view(-1)

    new_hists = torch.cat([hists_enc[range(bsz),hists_len-1, :int(d/2)], hists_enc[:, 0, int(d/2):]], dim=-1)

    return new_hists

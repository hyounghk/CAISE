import torch
import json
import os
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from decoders import GeneratorExtractor
from param import args
import numpy as np
from evaluate import LangEvaluator
import utils
from tqdm import tqdm
from results_stat import get_exact_match


class Speaker:
    def __init__(self, dataset):
        self.tok = dataset.tok
        self.feature_size = 2048

        ctx_size = 2048
        self.decoder = GeneratorExtractor(self.tok.vocab_size, ctx_size).cuda()

        self.optim = args.optimizer(list(self.decoder.parameters()),
                                    lr=args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.pad_id)

    def train(self, train_tuple, eval_tuple, num_epochs, rl=False):
        train_ds, train_tds, train_loader = train_tuple
        best_eval_score = -95
        best_minor_score = -95

        for epoch in range(num_epochs):
            print()
            iterator = tqdm(enumerate(train_loader), total=len(train_tds)//args.batch_size, unit="batch")
            word_accu = 0.
            for i, (uid, imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len) in iterator:
                api_gt = utils.cut_inst_with_leng(api_gt, api_leng)
                imgs, boxes, imgs_mask, concept, api_gt, hists = imgs.cuda(), boxes.cuda(), imgs_mask.cuda(), concept.cuda(), api_gt.cuda(), hists.cuda()
                self.optim.zero_grad()
                loss, batch_word_accu = self.teacher_forcing(imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len, train=True)
                word_accu += batch_word_accu
                iterator.set_postfix(loss=loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.)
                self.optim.step()
            word_accu /= (i + 1)  
            print("Epoch %d, Training Word Accuracy %0.4f" % (epoch, word_accu))

            if epoch % 1 == 0:
                print("Epoch %d" % epoch)

                scores, uid2pred = self.evaluate(eval_tuple)


                if scores > best_eval_score:
                    if scores > best_eval_score:
                        best_epoch = epoch
                        self.save("best_eval")
                        best_eval_score = scores

                print("scores:", scores)
                print("best_epoch:", best_epoch)
                print("best_score:", best_eval_score)



    def teacher_forcing(self, imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len, train=True):
        if train:
            self.decoder.train()
        else:
            self.decoder.eval()

        batch_size = imgs.size(0)
        _, hist_len, _ = hists.size()
        _, num_img, o_num, _ = concept.size()

        h_t = torch.zeros(1, batch_size, int(args.hid_dim)).cuda()
        c_t = torch.zeros(1, batch_size, int(args.hid_dim)).cuda()

        h_1 = torch.zeros(2, batch_size*hist_len, int(args.hid_dim/2)).cuda()
        c_1 = torch.zeros(2, batch_size*hist_len, int(args.hid_dim/2)).cuda()

        h_2 = torch.zeros(2, batch_size*num_img*o_num, int(args.hid_dim/2)).cuda()
        c_2 = torch.zeros(2, batch_size*num_img*o_num, int(args.hid_dim/2)).cuda()

        logits, h_t,c_t, h_1, c_1, h_2, c_2, _ = self.decoder(imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len, h_t,c_t,h_1, c_1, h_2, c_2,None)


        gold_probs = torch.gather(logits[:, :-1, :].contiguous().view(-1, self.tok.vocab_size), 1, api_gt[:, 1:].contiguous().view(-1,1))
        logit_log = -torch.log(gold_probs + 0.00000001)

        gt_mask = torch.zeros_like(api_gt[:, 1:])

        for i in range(batch_size):
            gt_mask[i,:api_leng[i]-1] = 1

        logit_log = logit_log.view(batch_size, -1) * gt_mask
        loss = logit_log.sum()/gt_mask.sum()

        _, predict_words = logits.max(2)   # B, l
        correct = (predict_words[:, :-1] == api_gt[:, 1:])
        word_accu = correct.sum().item() / (api_gt != self.tok.pad_id).sum().item()

        return loss, word_accu

    def infer_batch(self, imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len, sampling=True, train=False):

        if train:
            self.decoder.train()
        else:
            self.decoder.eval()

        batch_size = imgs.size(0)

        _, hist_len, _ = hists.size()
        _, num_img, o_num, _ = concept.size()

        word = np.ones(batch_size, np.int64) * self.tok.bos_id    
        words = [word]
        h_t = torch.zeros(1, batch_size, args.hid_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.hid_dim).cuda()

        h_1 = torch.zeros(2, batch_size*hist_len, int(args.hid_dim/2)).cuda()
        c_1 = torch.zeros(2, batch_size*hist_len, int(args.hid_dim/2)).cuda()

        h_2 = torch.zeros(2, batch_size*num_img*o_num, int(args.hid_dim/2)).cuda()
        c_2 = torch.zeros(2, batch_size*num_img*o_num, int(args.hid_dim/2)).cuda()

        ended = np.zeros(batch_size, np.bool)
        word = torch.from_numpy(word).view(-1, 1).cuda()
        log_probs = []
        hiddens = []
        entropies = []
        gates = []
        device = torch.device('cpu')
        for i in range(9):

            logits, h_t, c_t, h_1, c_1, h_2, c_2, gate = self.decoder(imgs, boxes, img_leng, imgs_mask, concept, concept_leng, word, api_leng, hists, hists_len, h_t,c_t, h_1, c_1, h_2, c_2, None)      # Decode, logits: (b, 1, vocab_size)

            logits = logits.squeeze(1)                                       
            logits[:, self.tok.unk_id] = -float("inf")                      
            if sampling:
                probs = F.softmax(logits, -1)
                m = Categorical(probs)
                word = m.sample()
                if train:
                    log_probs.append(m.log_prob(word))
                    hiddens.append(h_t)
                    entropies.append(m.entropy())
            else:
                values, word = logits.max(1)

            cpu_word = word.to(device).numpy()
            cpu_word[ended] = self.tok.pad_id
            words.append(cpu_word)
            gates.append(gate.squeeze(1).detach().to(device).numpy())

            word = word.view(-1, 1)

            ended = np.logical_or(ended, cpu_word == self.tok.eos_id)
            if ended.all():
                break
        if train:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(entropies, 1), \
                   torch.stack(hiddens, 1)
        else:
            return np.stack(words, 1), np.stack(gates, 1)       

    def evaluate(self, eval_tuple, split="", iters=-1):
        dataset, th_dset, dataloader = eval_tuple
        evaluator = LangEvaluator(dataset)

        all_insts = []
        all_gts = []
        all_gates = []
        uids = []
        word_accu = 0.
        for i, (uid, imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len) in enumerate(dataloader):
            if i == iters:
                break
            imgs, boxes, imgs_mask, concept, hists = imgs.cuda(),  boxes.cuda(), imgs_mask.cuda(), concept.cuda(), hists.cuda()

            infer_inst, gates = self.infer_batch(imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len, sampling=False, train=False)

            gates = gates[:,:-1,:]
            all_insts.extend(infer_inst)
            all_gates.extend(gates)
            all_gts.extend(api_gt.cpu().numpy())
            uids.extend(uid)

 
        for _ in range(3):
            import random
            i = random.randint(0, len(all_gts)-1)
            print('GT:   ' + self.tok.decode(self.tok.shrink(all_gts[i])))
            print('Pred: ' + self.tok.decode(self.tok.shrink(all_insts[i])))


        assert len(uids) == len(all_insts) == len(all_gts)

        uid2pred = {uid: (self.tok.decode(self.tok.shrink(pred)), gate)
                    for (uid, pred, gate) in zip(uids, all_insts, all_gates)}  
        scores = self.get_scores(uid2pred, evaluator.uid2ref, split=split)
        

        return scores, uid2pred

    def get_scores(self, uid2pred, uid2ref, split=""):
        gen_captions = {}
        for uid, (pred, gate) in uid2pred.items():
            new_gate = []
            for g in gate:
                new_gate_inner = []
                for f in g:
                    new_gate_inner.append(round(f.item(), 2))
                new_gate.append(new_gate_inner)

            gen_captions[uid] = (pred, uid2ref[uid], new_gate)

        if split != "":
            with open("best_" + split + ".json", 'w') as jf:
                json.dump(gen_captions, jf, indent=4, sort_keys=True)

        return get_exact_match(gen_captions)

    def save(self, name):
        decoder_path = os.path.join(self.output, '%s_dec.pth' % name)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, path):
        print("Load Speaker from %s" % path)
        dec_path = os.path.join(path + "_dec.pth")
        enc_state_dict = torch.load(enc_path)
        dec_state_dict = torch.load(dec_path)
        self.decoder.load_state_dict(dec_state_dict)

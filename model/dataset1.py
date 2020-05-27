#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import torch
from torch.utils.data import Dataset
from .text import BPEVocab
import os

class FacebookDataset(Dataset):
    @staticmethod
    def parse_data(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            i = 0
            j = 0
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                space_idx = line.find(' ')
                if space_idx == -1:
                    dialog_idx = int(line)
                else:
                    dialog_idx = int(line[:space_idx])

                if int(dialog_idx) == 1:
                    data.append({'persona_info': [], 'dialog': []})
                    i += 1
                dialog_line = line[space_idx + 1:].split('\t')
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[0].startswith('your persona:'):
                    persona_info = dialog_line[0].replace('your persona: ', '')
                    data[-1]['persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])
                    j += 1
            print("train_turn:",j)
            print("train_dialogue:",i)
            return data
    
    @staticmethod
    def make_dataset(data, vocab, max_lengths):
        dataset = []
        filename = 'data_test.txt'
        i = 0
        with open(filename,'a+') as f:
            for chat in data:
                i += 1
                persona_info = [vocab.string2ids(s) for s in chat['persona_info']]
                dialog = [vocab.string2ids(s) for s in chat['dialog']]

                if len(dialog) % 2 == 1:
                    dialog = dialog[:-1]
                if i > 689: 
                    persona_info_ent = [vocab.string2ent(s) for s in chat['persona_info']]
                    dialog_ent = [vocab.string2ent(s) for s in chat['dialog']]
                    dataset.append((persona_info, dialog, persona_info_ent, dialog_ent))
                    print(dataset[-1].__str__())
                    f.write(dataset[-1].__str__()+'\n')
        f.close()
        return dataset
    '''
    @staticmethod
    def make_dataset(data, vocab, max_lengths):
        dataset = []
        for chat in data:
            persona_info = [vocab.string2ids(s) for s in chat['persona_info']]
            dialog = [vocab.string2ids(s) for s in chat['dialog']]
            if len(dialog) % 2 == 1:
                dialog = dialog[:-1]   
            dataset.append((persona_info, dialog))
        return dataset
    '''
    def __init__(self, paths, vocab, max_lengths=2048, min_infos=2):
        assert min_infos > 0             

        if isinstance(paths, str):
            paths = [paths]

        self.entity2id = {}
        with open("kg_embed/entity2id.txt") as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                self.entity2id[qid] = int(eid)

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.min_infos = min_infos

        
        self.data = []
        '''
        with open(paths[0]) as fin:
            fin.readline()
            for line in fin:
                self.data.append(tuple(eval(line)))
        '''
        parsed_data = sum([FacebookDataset.parse_data(path) for path in paths], [])
        self.data = FacebookDataset.make_dataset(parsed_data, vocab, max_lengths)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_info, dialog, persona_info_ent, dialog_ent = self.data[idx]
        if len(persona_info):
            temp = list(zip(persona_info,persona_info_ent))
            n_info_samples = max(self.min_infos, random.randint(1, len(persona_info)))
            n_info_samples = min(n_info_samples, len(persona_info))
            #persona_info = random.sample(persona_info, n_info_samples)
            temp = random.sample(temp,n_info_samples)
            #random.shuffle(persona_info)
            random.shuffle(temp)
            persona_info[:],persona_info_ent[:] = zip(*temp)

            persona_info = sum(persona_info, []) 
            persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + [self.vocab.info_eos_id]
            persona_info_ent = sum(persona_info_ent, [])
            info_ents = []
            info_ents_mask = []
            for ent in persona_info_ent:
                if ent != "unk" and ent in self.entity2id:
                    info_ents.append(self.entity2id[ent])
                    info_ents_mask.append(1)
                else:
                    info_ents.append(-1)
                    info_ents_mask.append(0)
            persona_info_ent = [-1] + info_ents[:self.max_lengths-2] + [-1]
            p_mask = [1] + info_ents_mask[:self.max_lengths-2] + [0]
        dialog_begin = 0
        dialog_end = random.randrange(2, len(dialog)+1, 2)

        h = []
        h_ent = []
        h_mask = []
        for i, ids in enumerate(dialog[dialog_begin:dialog_end-1], 1):

            ids_ent = dialog_ent[i-1]
            indexed_ents = []
            ent_mask = []
            for ent in ids_ent:
                if ent != "unk" and ent in self.entity2id:
                    indexed_ents.append(self.entity2id[ent])
                    ent_mask.append(1)
                else:
                    indexed_ents.append(-1)
                    ent_mask.append(0)

            if i % 2 == 1:
                ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
            else:
                ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]

            indexed_ents = [-1] + indexed_ents + [-1]
            ent_mask = [0] + ent_mask + [0]
            h_mask.extend(ent_mask)
            h_ent.extend(indexed_ents)
            h.extend(ids)
        h = h[-self.max_lengths:]
        h_ent = h_ent[-self.max_lengths:]
        h_mask = h_mask[-self.max_lengths:]
        h_mask[0] = 1

        assert len(h) == len(h_ent)

        y = [self.vocab.bos_id] + dialog[dialog_end-1] + [self.vocab.eos_id]
        y = y[:self.max_lengths]

        return persona_info, h, y, persona_info_ent, p_mask, h_ent, h_mask

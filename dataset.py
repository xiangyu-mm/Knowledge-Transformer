import torch
import random

from model.dataset_forSeq import FacebookDataset
from config import get_model_config, get_trainer_config


def main():
    model_config = get_model_config()
    trainer_config = get_trainer_config()


    #train_dataset = FacebookDataset('./datasets/ConvAI2/train_self_revised_no_cands.txt')
    test_dataset = FacebookDataset('./datasets/ConvAI2/valid_self_original_no_cands.txt')

   
if __name__ == '__main__':
    main()


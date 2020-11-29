import pickle
import torch
from tqdm import tqdm
from pyvi import ViTokenizer
from os import listdir
from os.path import join
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from src.miscc.config import cfg, cfg_from_file


class TextEmbedding:
    def __init__(self):
        # Load the model in fairseq
        MODEL_DIR = '/usr/local/software/pretrained-models/PhoBERT_base_fairseq'
        __checkpoint_file = join(MODEL_DIR, 'model.pt')
        self.__phoBERT = RobertaModel.from_pretrained(MODEL_DIR,
                                                      checkpoint_file=__checkpoint_file)
        self.__phoBERT.eval()  # disable dropout (or leave in train mode to finetune

        # Khởi tạo Byte Pair Encoding cho PhoBERT
        class BPE():
            bpe_codes = '/usr/local/software/pretrained-models/PhoBERT_base_fairseq/bpe.codes'

        args = BPE()
        self.__phoBERT.bpe = fastBPE(args)  # Incorporate the BPE encoder into PhoBERT

    def embed_single_text(self, text, tokenized=False):
        if tokenized is False:
            text = ViTokenizer.tokenize(text)
        __encoded = self.__phoBERT.encode(text)
        last_layer_features = self.__phoBERT.extract_features(__encoded)
        feature = torch.mean(last_layer_features[0], dim=0)
        return feature.detach().cpu().numpy()

    def embed_text(self, tokenized_texts, embedding_file=None):
        features = []
        for t in tqdm(tokenized_texts):
            features.append(self.embed_single_text(t, tokenized=True))

        if embedding_file is not None:
            with open(embedding_file, 'wb') as f:
                pickle.dump(features, f)
            f.close()
        return features

    def embed_all_text_data(self, embedding_file=None):
        text_dir = join(cfg.DATA_DIR, 'text')
        descs = []
        for sub_dir in listdir(text_dir):
            for txt in listdir(join(text_dir, sub_dir)):
                txt_file_path = join(text_dir, sub_dir, txt)
                lines = []
                with open(txt_file_path, 'r') as f:
                    for line in f:
                        lines.append(line)
                desc = ViTokenizer.tokenize('.'.join(lines))
                descs.append((int(txt.split('.')[0]), desc))
        descs.sort(key=lambda tup: tup[0])
        sorted_desc = map(lambda tup: tup[1], descs)

        embedding_file = join(text_dir, 'embedding.pkl') if embedding_file is None else embedding_file
        self.embed_text(sorted_desc, embedding_file=embedding_file)


if __name__ == '__main__':
    cfg_from_file('/home/sondn/DIY/StackGAN-Pytorch/src/cfg/vn-celeb_eval.yml')
    embed = TextEmbedding()
    embed.embed_all_text_data('/home/sondn/DIY/StackGAN-Pytorch/data/emb.pkl')

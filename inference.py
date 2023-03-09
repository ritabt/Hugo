# Ignore warnings
import sys, os, warnings, pdb
warnings.filterwarnings('ignore')
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
blockPrint()

import torch
import torch.nn as nn

import numpy as np

import json

import captioning.utils.opts as opts
import captioning.models as models
import captioning.utils.misc as utils

import pytorch_lightning as pl

import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from timm.models.vision_transformer import resize_pos_embed
import openai

# Global Vars
device = 'cpu' #@param ["cuda", "cpu"] {allow-input: true}
reward = 'clips_grammar' #@param ["mle", "cider", "clips", "cider_clips", "clips_grammar"] {allow-input: true}


# Checkpoint class
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_keyboard_interrupt(self, trainer, pl_module):
        # Save model when keyboard interrupt
        filepath = os.path.join(self.dirpath, self.prefix + 'interrupt.ckpt')
        self._save_model(filepath)


# Device and Model Configuration
def load_models():
    if reward == 'mle':
        cfg = f'./configs/phase1/clipRN50_{reward}.yml'
    else:
        cfg = f'./configs/phase2/clipRN50_{reward}.yml'

    opt = opts.parse_opt(parse=False, cfg=cfg)

    # Load Vocabulary
    dict_json = json.load(open('./data/cocotalk.json'))

    ix_to_word = dict_json['ix_to_word']
    vocab_size = len(ix_to_word)

    seq_length = 1

    opt.vocab_size = vocab_size
    opt.seq_length = seq_length

    # Load Model Checkpoint
    opt.batch_size = 1
    opt.vocab = ix_to_word
    # opt.use_grammar = False

    model = models.setup(opt)
    del opt.vocab

    ckpt_path = opt.checkpoint_path + '-last.ckpt'

    raw_state_dict = torch.load(
        ckpt_path,
        map_location=device)

    strict = True

    state_dict = raw_state_dict['state_dict']

    if '_vocab' in state_dict:
        model.vocab = utils.deserialize(state_dict['_vocab'])
        del state_dict['_vocab']
    elif strict:
        raise KeyError
    if '_opt' in state_dict:
        saved_model_opt = utils.deserialize(state_dict['_opt'])
        del state_dict['_opt']
        # Make sure the saved opt is compatible with the curren topt
        need_be_same = ["caption_model",
                        "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            if getattr(saved_model_opt, checkme) in ['updown', 'topdown'] and \
                    getattr(opt, checkme) in ['updown', 'topdown']:
                continue
            assert getattr(saved_model_opt, checkme) == getattr(
                opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
    elif strict:
        raise KeyError
    res = model.load_state_dict(state_dict, strict)

    model = model.to(device)
    model.eval()

    # Load CLIP Image Encoder
    clip_model, clip_transform = clip.load("RN50", jit=False, device=device)

    preprocess = Compose([
        Resize((448, 448), interpolation=Image.BICUBIC),
        CenterCrop((448, 448)),
        ToTensor()
    ])

    image_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(device).reshape(3, 1, 1)
    image_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(device).reshape(3, 1, 1)

    num_patches = 196 #600 * 1000 // 32 // 32
    pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, clip_model.visual.attnpool.positional_embedding.shape[-1],  device=device),)
    pos_embed.weight = resize_pos_embed(clip_model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed)
    clip_model.visual.attnpool.positional_embedding = pos_embed

    return model, clip_model, opt, (preprocess, image_mean, image_std)


def caption_image(img_path, args, to_print=False):
    model, clip_model, opt, preprocess_info = args
    preprocess, image_mean, image_std = preprocess_info

    # Extract Visual Features
    with torch.no_grad():
        image = preprocess(Image.open( img_path ).convert("RGB"))
        image = torch.tensor(np.stack([image])).to(device)
        image -= image_mean
        image /= image_std
        
        tmp_att, tmp_fc = clip_model.encode_image(image)
        tmp_att = tmp_att[0].permute(1, 2, 0)
        tmp_fc = tmp_fc[0]
        
        att_feat = tmp_att
        fc_feat = tmp_fc

    # Generate Caption
    # Inference configurations
    eval_kwargs = {}
    eval_kwargs.update(vars(opt))

    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)

    # dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)

    with torch.no_grad():
        fc_feats = torch.zeros((1,0)).to(device)
        att_feats = att_feat.view(1, 196, 2048).float().to(device)
        att_masks = None

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp_eval_kwargs = eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 1})
        seq, seq_logprobs = model(
            fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        seq = seq.data

        sents = utils.decode_sequence(model.vocab, seq)

    if to_print:
        enablePrint()
        print(sents)
        blockPrint()
    return sents

def run_inference(images_path, save_path=None, to_print=False):
    if to_print:
        enablePrint()
        print("______________Start Generating Captions______________")
        blockPrint()
    if save_path is None:
        save_path = images_path + '/captions.json'
    args = load_models()
    output = {}
    i = 0
    for image in os.listdir(images_path):
        is_image = image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg")
        if is_image:
            img_path = images_path + '/' + image
            img_caption = caption_image(img_path, args, to_print=to_print)
            output[i] = {'image': image, 'caption': img_caption}
            i += 1            

    with open(save_path, "w") as outfile:
        json.dump(output, outfile)

    if to_print:
        enablePrint()
        print("______________Done Generating Captions______________")
        blockPrint()
    return output

def call_chat_gpt(prompt, to_print=False):
    if to_print:
        enablePrint()
        print("____________Start Generating ChatGPT Story___________")
        blockPrint()
    openai.api_key = ''
    model_engine = 'text-davinci-003'
    # prompt = "Tell me a short story about a yellow stuffed animal wearing a hat with the city skyline behind it"
    completion = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1,stop=None,temperature=0.7)
    message = completion.choices[0].text
    if to_print:
        enablePrint()
        print("INPUT PROMPT: ", prompt)
        print("CHATGPT OUTPUT: ", message)
        blockPrint()
    if to_print:
        enablePrint()
        print("____________Done Generating ChatGPT Story___________")
        blockPrint()
    return message

def make_prompt(captions):
    prompt = "tell me a short story about"
    for k in captions:
        if k>0:
            prompt += " and"
        prompt += " " + captions[k]['caption'][0]
    return prompt

useGPT = True


images_path = str(sys.argv[1])
captions = run_inference(images_path, to_print=True)
if useGPT:
    prompt = make_prompt(captions)
    call_chat_gpt(prompt, to_print=True)



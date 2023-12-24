# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com

import yaml
import numpy as np
from scipy import linalg
from tqdm import tqdm
from src.models.inception import InceptionV3

import torch
from torch.autograd import Variable
from torch import distributed as dist
from torch.nn.functional import adaptive_avg_pool2d, normalize
from torchvision import transforms
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy.random as random
import pickle


def get_config(args):
    config_file = args.config
    with open(config_file, 'r', encoding='utf8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if args.mode:
        configs['mode'] = args.mode

    if configs['mode'] == 'train':
        if args.epochs:
            configs['epochs'] = args.epochs

        if args.gpus:
            configs['gpus'] = args.gpus

        if args.output:
            configs['output'] = args.output

        if args.batch_size:
            configs['batch_size'] = args.batch_size

        if args.resume_epoch:
            configs['resume_epoch'] = args.resume_epoch

        if args.resume_model_path:
            configs['resume_model_path'] = args.resume_model_path

        if args.sc is not None:
            configs['sc'] = args.sc

        if args.match_loss is not None:
            configs['loss']['match_loss'] = args.match_loss

        if args.clip_rate:
            configs['loss']['clip_rate'] = args.clip_rate
    elif configs['mode'] != 'other':
        configs['load_dir'] = args.load_dir
    #     configs['log'] = args.log
        # if args.batch_size:
        #     configs['batch_size'] = args.batch_size

    return configs


def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


def load_npz(path):
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s


# save and load models
def load_opt_weights(optimizer, weights):
    optimizer.load_state_dict(weights)
    return optimizer


def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()


# DDP utils
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def sort_sents(captions, caption_lens):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = Variable(captions).cuda()
    sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    return captions, sorted_cap_lens, sorted_cap_indices


def encode_tokens(text_encoder, caption, cap_lens):
    # encode text
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            hidden = text_encoder.init_hidden(caption.size(0))
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img


def load_model_weights(model, weights, multi_gpus, train=True):
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model


def generate_samples(noise, caption, model, sc=None):
    with torch.no_grad():
        fake = model(noise, caption)
        fake2 = fake
        if sc:
            fake2 = sc(fake, caption)
    return fake, fake2


def divide_pred(pred):
    # the prediction contains the intermediate outputs of multiscale GAN,
    # so it's usually a list
    if type(pred) == list:
        fake = []
        real = []
        for p in pred:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    else:
        fake = pred[:pred.size(0) // 2]
        real = pred[pred.size(0) // 2:]

    return fake, real


def transf_to_CLIP_input(inputs):
    device = inputs.device
    if len(inputs.size()) != 4:
        raise ValueError('Expect the (B, C, X, Y) tensor.')
    else:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]) \
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        var = torch.tensor([0.26862954, 0.26130258, 0.27577711]) \
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        inputs = torch.nn.functional.interpolate(inputs * 0.5 + 0.5, size=(224, 224))
        inputs = ((inputs + 1) * 0.5 - mean) / var
        return inputs


def compute_act_mean_std(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# compute FID
def compute_FID(mu1, mu2, sigma1, sigma2,eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return FID


def get_FID(dataloader, configs, inception, netG, netS, text_encoder, tokenize, device, lens):
    batch_size = configs['batch_size']

    norm = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize((299, 299)),
    ])
    n_act = 2048
    act1 = np.zeros((lens, n_act))
    act2 = np.zeros((lens, n_act))
    for i, batch in enumerate(dataloader):
        imgs, caption, keys = batch
        imgs = norm(imgs).to(device)
        batch_size_i = imgs.size()[0]
        activation = inception(imgs)[0].cpu().data.numpy().reshape(batch_size_i, -1)
        act1[i * batch_size: i * batch_size + batch_size_i] = activation

        sent_emb = tokenize(caption, truncate=True).to(device)
        sent_emb, words_embs = text_encoder(sent_emb)

        with torch.no_grad():
            if configs['truncation']:
                noise = truncated_noise(batch_size_i, configs['z_dim'], configs['trunc_rate'])
                noise = torch.tensor(noise, dtype=torch.float).to(device)
            else:
                noise = torch.randn(batch_size_i, configs['z_dim']).to(device)
            fake_imgs = netG(noise, sent_emb)
            if netS:
                fake_imgs = netS(fake_imgs, sent_emb)
            fake = norm(fake_imgs)
            pred = inception(fake)[0].cpu().data.numpy().reshape(batch_size_i, -1)
            act2[i * batch_size: i * batch_size + batch_size_i] = pred

    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)

    FID = compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return FID


def get_batch_FID(batch, configs, inception, netG, netS, text_encoder, tokenize, device, number=5):
    batch_size = configs['batch_size']

    norm = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize((299, 299)),
    ])
    n_act = 2048

    imgs, caption, keys = batch

    batch_size_i = len(imgs)

    imgs = norm(imgs).to(device)
    act1 = np.zeros((batch_size_i * number, n_act))
    act2 = np.zeros((batch_size_i * number, n_act))

    for i in range(number):
        # for i in range(number):
        activation = inception(imgs)[0].cpu().data.numpy().reshape(-1, n_act)
        act1[i * batch_size_i: i * batch_size_i + batch_size_i] = activation

        sent_emb = tokenize(caption, truncate=True).to(device)
        sent_emb, words_embs = text_encoder(sent_emb)

        with torch.no_grad():
            if configs['truncation']:
                noise = truncated_noise(batch_size_i, configs['z_dim'], configs['trunc_rate'])
                noise = torch.tensor(noise, dtype=torch.float).to(device)
            else:
                noise = torch.randn(batch_size_i, configs['z_dim']).to(device)
            fake_imgs = netG(noise, sent_emb)
            if netS:
                fake_imgs = netS(fake_imgs, sent_emb)
            fake = norm(fake_imgs)
            pred = inception(fake)[0].cpu().data.numpy().reshape(-1, n_act)
            act2[i * batch_size: i * batch_size + batch_size_i] = pred

    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)

    FID = compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return FID



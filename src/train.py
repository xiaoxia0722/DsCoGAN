# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com

from src.models import *
from src.dataset_img import *
from src.losses import *

import time

import torch
import torchvision
from thop import profile
from torchvision.utils import make_grid
from torchvision import transforms
import clip


class Trainer:
    def __init__(self, configs, device, logger):
        self.configs = configs
        self.device = device
        self.logger = logger
        self.image_encoder = None
        self.text_encoder = None
        self.netG = None
        self.netD = None
        self.netC = None
        self.optimizerG = None
        self.optimizerD = None
        self.m1 = None
        self.s1 = None
        self.netS = None
        self.inception = None
        self.transform = transforms.Resize(224)
        self.norm = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
            transforms.Resize((299, 299)),
        ])
        logger.info("初始化模型...")
        self.init_model()

        logger.info("初始化优化器...")
        self.init_opt()

        self.get_ms()

        self.criterionFeat = nn.L1Loss()

    def init_model(self):
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_encoder = CLIP_TXT_ENCODER(self.clip)
        self.image_encoder = CLIP_Image_Encoder(self.clip)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        if self.configs['sc']:
            self.netS = SemanticCorrection(self.clip, self.configs['model']['embedding_dim'], self.configs['ch_size'])
            self.netS.to(self.device)

        # GAN models
        self.netG = NetG(self.configs['nf'], self.configs['z_dim'], self.configs['model']['embedding_dim'], self.configs['imsize'], self.configs['ch_size']).to(self.device)
        self.netD = NetD(self.configs['nf'], self.configs['imsize'], self.configs['ch_size']).to(self.device)
        self.netC = NetC(self.configs['nf'], self.configs['model']['embedding_dim']).to(self.device)

        if self.configs['many_gpus']:
            self.netG = nn.parallel.DistributedDataParallel(self.netG, broadcast_buffers=False, device_ids=self.configs['local_rank'], output_device=self.configs['local_rank'])
            self.netD = nn.parallel.DistributedDataParallel(self.netD, broadcast_buffers=False, device_ids=self.configs['local_rank'], output_device=self.configs['local_rank'])
            self.netC = nn.parallel.DistributedDataParallel(self.netC, broadcast_buffers=False, device_ids=self.configs['local_rank'], output_device=self.configs['local_rank'])

        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx])
        self.inception.to(self.device)
        self.inception.eval()

    def init_opt(self):
        params_G = list(self.netG.parameters())
        if self.netS:
            params_G += list(self.netS.parameters())
        self.optimizerG = torch.optim.Adam(params_G, lr=self.configs['generator']['lr'], betas=(self.configs['generator']['beta1'], self.configs['generator']['beta2']))
        params_D = list(self.netD.parameters()) + list(self.netC.parameters())
        self.optimizerD = torch.optim.Adam(params_D, lr=self.configs['discriminator']['lr'], betas=(self.configs['discriminator']['beta1'], self.configs['discriminator']['beta2']))

    def get_ms(self):
        self.m1, self.s1 = load_npz(self.configs['npz_path'])

    def get_ms2(self, dataloader):
        # prepare Inception V3
        self.inception.eval()
        self.netG.eval()
        dl_length = dataloader.__len__()
        imgs_num = dl_length * self.configs['batch_size'] * self.configs['sample_times']
        pred_arr = np.empty((imgs_num, 2048))
        loop = tqdm(total=int(dl_length * self.configs['sample_times']))
        for time in range(self.configs['sample_times']):
            for i, data in enumerate(dataloader):
                start = i * self.configs['batch_size'] + time * dl_length * self.configs['batch_size']
                end = start + len(data[0])
                imgs, caption, keys = data
                with torch.no_grad():
                    fake = self.norm(imgs).to(self.device)
                    pred = self.inception(fake)[0]
                    if pred.shape[2] != 1 or pred.shape[3] != 1:
                        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                    pred_all = pred.squeeze(-1).squeeze(-1)
                    pred_arr[start:end] = pred_all.cpu().data.numpy()
        loop.close()
        self.m1 = np.mean(pred_arr, axis=0)
        self.s1 = np.cov(pred_arr, rowvar=False)

    def train(self, epoch, train_loader):
        lens = len(train_loader)
        start = time.time()
        g_losses = []
        g1_losses = []
        g2_losses = []
        d_losses = []
        d_fake = []
        d_real = []
        ma_gp = []
        match_scores = []
        for step, batch in enumerate(train_loader):
            imgs, caption, keys = batch
            sent_token = clip.tokenize(caption, truncate=True)

            imgs = imgs.to(self.device).requires_grad_()
            sent_token = sent_token.to(self.device)

            sent_emb, words_embs = self.text_encoder(sent_token)

            sent_emb = sent_emb.requires_grad_()

            # 预测真实图片
            real_feature = self.netD(imgs)
            pred_real, errD_real = predict_loss(self.netC, real_feature, sent_emb, negtive=False)
            mis_features = torch.cat((real_feature[1:], real_feature[0:1]), dim=0)
            _, errD_mis = predict_loss(self.netC, mis_features, sent_emb, negtive=True)

            # 鉴别假图片
            noise = torch.randn(self.configs['batch_size'], self.configs['z_dim']).to(self.device)
            fake = self.netG(noise, sent_emb)
            fake_feature = self.netD(fake.detach())
            _, errD_fake = predict_loss(self.netC, fake_feature, sent_emb, negtive=True)

            errD_fake2 = 0
            if self.netS:
                fake2 = self.netS(fake, sent_emb)
                fake2_feature = self.netD(fake2.detach())
                _, errD_fake2 = predict_loss(self.netC, fake2_feature, sent_emb, negtive=True)

            # MA-GP
            errD_magp = MA_GP(imgs, sent_emb, pred_real)
            if self.netS:
                # 整个D的loss
                err_D = errD_real + (errD_fake + errD_mis + errD_fake2) / 3.0 + errD_magp
            else:
                err_D = errD_real + (errD_fake + errD_mis) / 2.0 + errD_magp

            # 更新D
            self.optimizerD.zero_grad()
            err_D.backward()
            self.optimizerD.step()

            # 更新G
            fake_features = self.netD(fake)
            output = self.netC(fake_features, sent_emb)
            clip_fake, fake_img_embed = self.image_encoder(fake)
            errG = -output.mean()
            output2 = torch.zeros(1)
            if self.netS:
                fake2_features = self.netD(fake2)
                output2 = self.netC(fake2_features, sent_emb)
                errG -= output2.mean()
                clip_fake2, fake_img_embed2 = self.image_encoder(fake2)
            match_score = torch.zeros(1)
            if self.configs['loss']['match_loss']:
                if self.netS:
                    match_score = self.get_clip_match(fake_img_embed2, sent_emb)
                    errG += match_score
                else:
                    match_score = self.get_clip_match(fake_img_embed, sent_emb)
                    errG += match_score
            self.optimizerG.zero_grad()
            errG.backward()
            self.optimizerG.step()

            if step % self.configs['print_step'] == 0:
                end = time.time()
                all_time = end - start
                self.logger.info("epoch:{}[step:{}/{}] => g_loss:{}, g_loss1:{}, g_loss2:{}, d_loss:{}, MA-GP:{}, output:{}, match_score:{}, d_real:{}, d_fake:{}, errD_mis:{}, 当前epoch耗时为:{}min, 预计总耗时为:{}min, 剩余耗时为:{}min".format(epoch, step, lens, errG.item(), -output.mean().item(), -output2.mean().item(), err_D.item(), errD_magp.item(), -output.mean(), match_score.item(), errD_real.item(), errD_fake.item(), errD_mis.item(), all_time / 60, all_time / (step + 1) * lens / 60, all_time / (step + 1) * (lens - step - 1) / 60))
                g_losses.append(errG.item())
                g1_losses.append(-output.mean().item())
                g2_losses.append(-output2.mean().item())
                d_losses.append(err_D.item())
                ma_gp.append(errD_magp.item())
                match_scores.append(match_score.item())
                d_fake.append(errD_fake.item())
                d_real.append(errD_real.item())
        return {
            'g_losses': g_losses,
            'g1_losses': g1_losses,
            'g2_losses': g2_losses,
            'd_losses': d_losses,
            'd_fake': d_fake,
            'd_real': d_real,
            'ma_gp': ma_gp,
            'match_score': match_scores,
            'time': time.time() - start
        }

    def val(self, epoch, val_dataloader):
        torch.cuda.empty_cache()
        fid = self.calculate_fid(val_dataloader, epoch)
        torch.cuda.empty_cache()
        clip_score = 0
        clip_score = self.get_clip_score(val_dataloader)
        return fid, clip_score

    def load_model_opt(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.netG = load_model_weights(self.netG, checkpoint['model']['netG'], self.configs['many_gpus'])
        self.netD = load_model_weights(self.netD, checkpoint['model']['netD'], self.configs['many_gpus'])
        self.netC = load_model_weights(self.netC, checkpoint['model']['netC'], self.configs['many_gpus'])
        self.optimizerG = load_opt_weights(self.optimizerG, checkpoint['optimizers']['optimizer_G'])
        self.optimizerD = load_opt_weights(self.optimizerD, checkpoint['optimizers']['optimizer_D'])

        if self.netS and 'netS' in checkpoint['model']:
            self.netS = load_model_weights(self.netS, checkpoint['model']['netS'], self.configs['many_gpus'])

    def save_model(self, mode, epoch):
        state = {'model': {'netG': self.netG.state_dict(), 'netD': self.netD.state_dict(), 'netC': self.netC.state_dict(), 'netS': self.netS.state_dict()},
                 'optimizers': {'optimizer_G': self.optimizerG.state_dict(), 'optimizer_D': self.optimizerD.state_dict()},
                 'epoch': epoch}
        if mode == 'epoch':
            torch.save(state, '%s/state_epoch_%03d.pth' % (self.configs['output'], epoch))
        else:
            torch.save(state, '%s/state_%s.pth' % (self.configs['output'], mode))

    def save_img(self, noise, sent, epoch, img_save_dir, writer):
        if not writer and not img_save_dir:
            return
        fixed_results, fixed_results2 = generate_samples(noise, sent, self.netG, self.netS)
        if writer:
            fixed_grid = make_grid(fixed_results.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('fixed results', fixed_grid, epoch)

            fixed_grid2 = make_grid(fixed_results2.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('fixed results2', fixed_grid2, epoch)
        if img_save_dir:
            img_name = 'samples_epoch_%03d.png' % (epoch)
            img_save_path = os.path.join(img_save_dir, img_name)
            torchvision.utils.save_image(fixed_results.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)

            img_name = 'samples_epoch_%03d_sc.png' % (epoch)
            img_save_path = os.path.join(img_save_dir, img_name)
            torchvision.utils.save_image(fixed_results2.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)

    def get_one_batch_data(self, dataloader):
        data = next(iter(dataloader))
        imgs, caption, keys, = data
        sent_emb = clip.tokenize(caption, truncate=True)

        imgs = imgs.to(self.device)
        sent_emb = sent_emb.to(self.device)

        sent_emb, words_embs = self.text_encoder(sent_emb)

        return imgs, sent_emb, caption

    def get_fix_data(self, train_dataloader, test_dataloader):
        fixed_image_train, fixed_sent_train, caption_train = self.get_one_batch_data(train_dataloader)
        fixed_image_test, fixed_sent_test, caption_test = self.get_one_batch_data(test_dataloader)
        fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
        fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
        fixed_caption = caption_train + caption_test
        if self.configs['truncation']:
            noise = truncated_noise(fixed_image.size(0), self.configs['z_dim'], self.configs['trunc_rate'])
            fixed_noise = torch.tensor(noise, dtype=torch.float).to(self.device)
        else:
            fixed_noise = torch.randn(fixed_image.size(0), self.configs['z_dim']).to(self.device)
        return fixed_image, fixed_sent, fixed_noise, fixed_caption

    def get_params(self):
        with torch.no_grad():
            caption = ['this is a small bird with a black head, white throat and neck, brown breast and belly and gray wings and tail, while its beak is too big for its body.']
            imgs = torch.randn((1, 3, 256, 256)).to(self.device)
            sent_emb = clip.tokenize(caption, truncate=True)

            sent_emb = sent_emb.to(self.device)
            sent_emb2, words_embs = self.text_encoder(sent_emb)
            noise = torch.randn(sent_emb2.size(0), self.configs['z_dim']).to(self.device)
            flops1, params1 = profile(self.text_encoder, inputs=(sent_emb))
            flops2, params2 = profile(self.netG, inputs=(noise, sent_emb2))
            flops3, params3 = 0, 0
            flops4, params4 = 0, 0
            if self.image_encoder:
                flops3, params3 = profile(self.image_encoder, inputs=(imgs, ))

            if self.netS:
                flops4, params4 = profile(self.netS, inputs=(imgs, sent_emb2))
        return {
            'text_encoder': {
                'flops': flops1,
                'params': params1
            },
            'generator': {
                'flops': flops2,
                'params': params2
            },
            'img_encoder': {
                'flops': flops3,
                'params': params3
            },
            'sc': {
                'flops': flops4,
                'params': params4
            },
            'all': {
                'flops': flops1 + flops2 + flops4,
                'params': params1 + params2 + params4
            }
        }

    def get_clip_match(self, fake_img_embed, sent_emb):
        score = -torch.cosine_similarity(fake_img_embed, sent_emb).mean() * self.configs['loss']['clip_rate']
        return score.float()

    def normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())

    def calculate_fid(self, dataloader, epoch):
        """ Calculates the FID """
        # prepare Inception V3
        self.inception.eval()
        self.netG.eval()
        dl_length = dataloader.__len__()
        imgs_num = dl_length * self.configs['batch_size'] * self.configs['sample_times']
        pred_arr = np.empty((imgs_num, 2048))
        loop = tqdm(total=int(dl_length * self.configs['sample_times']))
        for time in range(self.configs['sample_times']):
            for i, data in enumerate(dataloader):
                start = i * self.configs['batch_size'] + time * dl_length * self.configs['batch_size']
                end = start + len(data[0])
                ######################################################
                # (1) Prepare_data
                ######################################################

                imgs, caption, keys = data
                if self.configs['model']['encoder_type'] != 'clip':
                    sent_emb = self.tokenizer(caption, return_tensors='pt', max_length=self.configs['text']['words_num'],
                                              padding=True, truncation=True)
                else:
                    sent_emb = clip.tokenize(caption, truncate=True)

                sent_emb = sent_emb.to(self.device)
                if self.configs['model']['encoder_type'] != 'clip':
                    sent_emb = self.text_encoder(input_ids=sent_emb['input_ids'], attention_mask=sent_emb['attention_mask'])
                else:
                    sent_emb, words_embs = self.text_encoder(sent_emb)
                ######################################################
                # (2) Generate fake images
                ######################################################
                batch_size = sent_emb.size(0)
                self.netG.eval()
                with torch.no_grad():
                    if self.configs['truncation']:
                        noise = truncated_noise(batch_size, self.configs['z_dim'], self.configs['trunc_rate'])
                        noise = torch.tensor(noise, dtype=torch.float).to(self.device)
                    else:
                        noise = torch.randn(batch_size, self.configs['z_dim']).to(self.device)
                    fake_imgs = self.netG(noise, sent_emb)
                    if self.netS:
                        fake_imgs = self.netS(fake_imgs, sent_emb)
                    fake = self.norm(fake_imgs)
                    pred = self.inception(fake)[0]
                    if pred.shape[2] != 1 or pred.shape[3] != 1:
                        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                    pred_all = pred.squeeze(-1).squeeze(-1)
                    pred_arr[start:end] = pred_all.cpu().data.numpy()
                loop.update(1)
                loop.set_description(f'Evaluate Epoch [{epoch}/{self.configs["epochs"]}]')
                loop.set_postfix()
        loop.close()
        m2 = np.mean(pred_arr, axis=0)
        s2 = np.cov(pred_arr, rowvar=False)
        fid_value = calculate_frechet_distance(self.m1, self.s1, m2, s2)

        return fid_value

    def get_clip_score(self, dataloader):
        clip_match_scores1 = []
        clip_match_scores2 = []

        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()
            imgs, caption, keys = batch

            sent_token = clip.tokenize(caption, truncate=True)
            sent_token = sent_token.to(self.device)

            sent_emb, words_embs = self.text_encoder(sent_token)

            noise = torch.randn(len(sent_emb), self.configs['z_dim']).to(self.device)
            fake = self.netG(noise, sent_emb)

            fake2 = self.netS(fake, sent_emb)

            fake = transf_to_CLIP_input(fake)
            fake2 = transf_to_CLIP_input(fake2)

            fake_feature1 = self.clip.encode_image(fake)
            fake_feature2 = self.clip.encode_image(fake2)

            cs1 = torch.cosine_similarity(fake_feature1, sent_emb).mean().item()
            cs2 = torch.cosine_similarity(fake_feature2, sent_emb).mean().item()

            clip_match_scores1.append(cs1)
            clip_match_scores2.append(cs2)

            del fake
            del fake2

            del fake_feature1
            del fake_feature2

        return {
            'clip score1': sum(clip_match_scores1) / len(clip_match_scores1),
            'clip score2': sum(clip_match_scores2) / len(clip_match_scores2)
        }

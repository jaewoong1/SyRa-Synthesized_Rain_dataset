import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
import glob

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets, self.nets_ema = build_model(args)

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels

            inputs = next(fetcher)

            size = torch.empty(args.batch_size)
            l = torch.ones_like(size).type(torch.long).to('cuda')

            x_gt, x_rain = inputs.x_src[:, :, :, :256], inputs.x_src[:, :, :, 256:]
            y_gt, y_rain = inputs.x_ref[:, :, :, :256], inputs.x_ref[:, :, :, 256:]

            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_gt) if args.w_hpf > 0 else None

            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_gt, x_rain, l, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_gt, x_rain, l, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_gt, x_rain, y_gt, y_rain, l, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_gt, x_rain, y_gt, y_rain, l, masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)


    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        self._load_checkpoint(args.resume_iter)
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)
        os.makedirs(args.sample_dir, exist_ok=True)
        for i in tqdm(range(2)):
            utils.debug_image(nets_ema, args, inputs=inputs_val, step=i)
            inputs_val = next(fetcher_val)

    @torch.no_grad()
    def synthesis(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        self._load_checkpoint(args.resume_iter)
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        os.makedirs(args.out_dir, exist_ok=True)
        inputs_val = next(fetcher_val)
        path = os.path.join(args.rain_dir, args.data)
        file_list = os.listdir(path)
        for i in tqdm(range(int(len(file_list)/args.val_batch_size))):
            utils.synthesis_image(nets_ema, args, inputs=inputs_val, step=i)
            inputs_val = next(fetcher_val)



def compute_d_loss(nets, args, x_gt, x_rain, l, z_trg=None, masks=None):

    x_gt.requires_grad_()
    x_rain.requires_grad_()
    out = nets.discriminator(x_rain, l)
    loss_reg = r1_reg(out, x_rain)

    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, l)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_rain, l)

        x_fake = nets.generator(x_gt, s_trg, masks=masks)
    out2 = nets.discriminator(x_fake, l)

    # Rain streak-guided loss
    if args.SGD == 1:
        loss_real = adv_loss(out-(0.1*out2), 1)
        loss_fake = adv_loss(out2-(0.1*out), 0)
    else:
        loss_real = adv_loss(out, 1)
        loss_fake = adv_loss(out2, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_gt, x_rain, y_gt, y_rain, l, z_trgs=None, masks=None):
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs

    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, l)
    else:
        s_trg = nets.style_encoder(x_rain, l)

    x_fake = nets.generator(x_gt, s_trg, masks=masks)

    out = nets.discriminator(x_fake, l)

    loss_adv = adv_loss(out, 1)

    s_pred = nets.style_encoder(x_fake, l)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, l)
    else:
        s_trg2 = nets.style_encoder(y_rain, l)
    x_fake2 = nets.generator(x_gt, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()

    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    masks = nets.fan.get_heatmap(y_gt) if args.w_hpf > 0 else None

    x_fake3 = nets.generator(y_gt, s_trg2, masks=masks)
    loss_rain = torch.mean(torch.abs(x_fake3 - y_rain))

    loss_NPMI = mutual_info(s_trg, s_trg2)
    s = s_trg2.clone()
    s -= s

    x_ori = nets.generator(x_gt, s, masks=masks)
    loss_bias = torch.mean(torch.abs(x_ori - x_gt))

    loss = loss_adv + args.lambda_sty * loss_sty \
        + loss_rain + args.lambda_NPMI * loss_NPMI + args.gt * loss_bias - args.lambda_ds * loss_ds

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       rain=loss_rain.item(),
                       bias=loss_bias.item(),
                       MI=loss_NPMI.item())

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def mutual_info(x_out, x_tf_out, EPS=1e-10):
    _, k = x_out.size()

    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = torch.log(p_i_j/(p_j*p_i)) / (-torch.log(p_i_j))

    loss = loss.mean()

    return loss

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


"""Skip-WGANomaly training + testing
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lib.models.networks import NetD, weights_init, define_G, define_C, get_scheduler, NetG
from lib.visualizer import Visualizer
from lib.loss import l2_loss, w_loss
from lib.evaluate import roc
from lib.models.basemodel import BaseModel

#torch.autograd.set_detect_anomaly(True) 

class W_Skipganomaly(BaseModel):
    """GANomaly Class
    """
    @property
    def name(self): return 'w_skipganomaly'

    def __init__(self, opt, data=None):
        super(W_Skipganomaly, self).__init__(opt, data)
        ##
        

        # -- Misc attributes
        self.add_noise = True
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.steps = 0 # for training generator 1/5 times of discriminator
        self.alambda = 10 # default gradient penalty lambda

        ##
        # Create and initialize networks.
        self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal')
        self.netd = define_C(self.opt, norm='batch', use_sigmoid=False, init_type='normal') #netd represents the critic

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        if self.opt.verbose:
            print(self.netg)
            print(self.netd)

        ##
        # Loss Functions
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizers  = []
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
    #Gradient penalty function from Aladdin Persson
    ## 
    """

    Copyright (c) 2020 Aladdin Persson

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    
    def gradient_penalty(critic, real, fake, device="cpu"):
        print("real", real)
        print("real size", real.size())
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * alpha + fake * (1 - alpha)
        
        # Calculate critic scores
        mixed_scores = critic(interpolated_images)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm -1) ** 2)
        return gradient_penalty
        
    
    def forward(self):
        
        self.forward_g()
        self.forward_d()
        

    def forward_g(self):
        """ Forward propagate through netD
        """
        self.fake = self.netg(self.input + self.noise)

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.feat_real = self.netd(self.input) 
        self.feat_fake = self.netd(self.fake)

    def backward_g(self):
        """ Backpropagate netg
        """
        
        self.feat_fake = self.netd(self.fake.detach())
        
        #Wasserstein distance for generator
        self.err_g_adv = self.opt.w_adv * (-torch.mean(self.feat_fake))
        #Reconstruction loss
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)

        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat.detach() 
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        
        #Latent loss
        self.err_g_lat = self.opt.w_lat * self.l_lat(self.feat_fake, self.feat_real)
        
        
        #Obtaining gradient penalty
        BATCH_SIZE, C, H, W = self.input.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
        interpolated_images = self.input * alpha + self.fake * (1 - alpha)
        
            # Calculate critic scores
        mixed_scores = self.netd(interpolated_images)
            # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm -1) ** 2)
        
        # Discriminator loss based in w distance + gradient penalty 
        self.err_d_wloss = torch.mean(self.feat_fake) - torch.mean(self.feat_real) + self.alambda * gradient_penalty

        self.err_d = self.err_d_wloss 
        
        self.err_d.backward(retain_graph=True)
        
    def update_netg(self):
        """ Update Generator Network.
        """       
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    def update_netd(self):
        """ Update Critic Network.
        """       
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d < 1e-5: self.reinit_d() 
    ##
    def optimize_params(self):
        """ Optimize netD and netG  networks.
        """
        
        self.steps +=1
        
        self.forward()
        self.update_netd()
        
        if ((self.steps+1) % 5) == 0: # every 5 batches
            self.update_netg()   
    ##
    def test(self, plot_hist=False):
        """ Test Skip-WGANomaly model.

        Args:
            data ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                self.load_weights(is_best=True)

            self.opt.phase = 'test'

            scores = {}

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.features  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            print("   Testing %s" % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                # Forward - Pass
                self.set_input(data)
                self.fake = self.netg(self.input)
                
                #Display real and generated image comparison for CIFAR-10,for a normal and anomalous case
                if (i==1):
                    i_1 = np.random.randint(0,high=5)
                    self.visualizer.display_real_fake_airplane(self.input[i_1], self.fake[i_1])            
                elif (i==200): 
                    i_2 = np.random.randint(0,high=5)
                    self.visualizer.display_real_fake_airplane(self.input[i_2], self.fake[i_2], 8)

                self.feat_real = self.netd(self.input) 
                self.feat_fake = self.netd(self.fake) 

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9*rec + 0.1*lat

                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc(self.gt_labels, self.an_scores)
            
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            ##
            if True: 
                plt.ion()
                # Create data frame for scores and labels.
                scores['scores'] = self.an_scores.cpu() #added .cpu()
                scores['labels'] = self.gt_labels.cpu() #added .cpu()
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv("histogram.csv")

                # Filter normal and abnormal scores.
                abn_scr = hist.loc[hist.labels == 1]['scores']
                nrm_scr = hist.loc[hist.labels == 0]['scores']

                # Create figure and plot the distribution.
                fig, ax = plt.subplots(figsize=(4,4));
                sns.distplot(nrm_scr, label=r'Normal Scores')
                sns.distplot(abn_scr, label=r'Abnormal Scores')
                #print(scores['scores'])
                #print(scores['labels'])

                plt.legend()
                plt.yticks([])
                plt.xlabel(r'Anomaly Scores')
   
            ##
            # PLOT PERFORMANCE
            if self.opt.display_id > 0 and self.opt.phase == 'test':
                print("testing")
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            ##
            # RETURN
            return performance

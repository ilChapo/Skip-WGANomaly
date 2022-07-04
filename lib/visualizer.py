""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
import numpy as np
import torchvision.utils as vutils
from PIL import Image, ImageChops

##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, )

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_data2 = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)
        now  = time.strftime("%c")
        title = f'================ {now} ================\n'
        info  = f'{opt.abnormal_class}, {opt.nz}, {opt.w_adv}, {opt.w_con}, {opt.w_lat}\n'
        self.write_to_log_file(text=title + info)


    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=4
        )
        

    ##
    ## AFEGIT
    def plot_current_AUC(self, epoch, auc):
        """Plot current AUC.

        Args:
            epoch (int): Current epoch
            auc (float): Area under the ROC curve result
        """
        if not hasattr(self, 'plot_data2') or self.plot_data2 is None:
            self.plot_data2 = {'X': [], 'Y': []}
        self.plot_data2['X'].append(epoch)
        self.plot_data2['Y'].append(auc)
        self.vis.line(
            X=np.stack(self.plot_data2['X']),
            Y=np.array(self.plot_data2['Y']),
            opts={
                'title': self.opt.abnormal_class + ' as anomaly in ' + self.opt.model,
                #legend': self.plot_data2['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'AUC value',
                'linecolor': np.array([[245,99,230]]),
                'ytickmin': 0.5,
                'ytickmax': 1,
                'markers': True,
                'markersize': 5
                
            },
            win=5
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=5
        )

    ##
    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            batch_i (int): Current batch
            batch_n (int): Total Number of batches.
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def write_to_log_file(self, text):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % text)

    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)
        message += 'max AUC: %.3f' % best

        print(message)
        self.write_to_log_file(text=message)

    def display_current_images(self, reals, fakes, fixed):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        #fixed = self.normalize(fixed.cpu().numpy()) 

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        #self.vis.images(fixed, win=3, opts={'title': 'Fixed'}) 
    
    def display_real_fake_airplane(self, real, fake, window = 6):
        """ Display pair real-generated image in visdom.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            window (int): window of visdom where to display the real image
        """
        real = self.normalize(real.cpu().numpy())
        fake = self.normalize(fake.cpu().numpy())
        
        self.vis.images(real, win=window, opts={'title': 'REAL IMG'})
        self.vis.images(fake, win=window+1, opts={'title': 'FAKE IMG'})
        
        #print(real.shape)
        #print(real.type())
        #print(real.size())
 
        #real_img = Image.fromarray(real.astype('uint8'), 'RGB')
        #fake_img = Image.fromarray(fake.astype('uint8'), 'RGB')
        #self.vis.images(ImageChops.difference(real_img, fake_img), win= window+2, opts={'title': 'DIFF GENERATED IMG'})


    def save_current_images(self, epoch, reals, fakes, fixed):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        vutils.save_image(reals, '%s/reals.png' % self.img_dir, normalize=True)
        vutils.save_image(fakes, '%s/fakes.png' % self.img_dir, normalize=True)
        vutils.save_image(fixed, '%s/fixed_fakes_%03d.png' %(self.img_dir, epoch+1), normalize=True)

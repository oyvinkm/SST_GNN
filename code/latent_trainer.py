import os
import json
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from model.utility import ShiftDistribution, make_noise, make_interpolation_chart, fig_to_image
from model.utility import MeanTracker, DeformatorType
from matplotlib import pyplot as plt
from utils.visualization import plot_loss
from loguru import logger


class Params(object):
    """Sets up parameters for the latenttrainer"""
    def __init__(self, **kwargs):
        self.shift_scale = 6.0
        self.min_shift = 0.5
        self.shift_distribution = ShiftDistribution.UNIFORM

        self.deformator_lr = 0.0001
        self.shift_predictor_lr = 0.0001
        self.n_steps = int(3e+3)
        self.batch_size = 8

        self.directions_count = None
        self.max_latent_dim = None

        self.label_weight = 1.0
        self.shift_weight = 0.25

        self.steps_per_log = 10
        self.steps_per_save = 10000
        self.steps_per_img_log = 1000
        self.steps_per_backup = 1000

        self.truncation = None

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


class Trainer(object):
    def __init__(self, params=Params(), out_dir='', verbose=False, device='cpu'):
        if verbose:
            print('Trainer inited with:\n{}'.format(str(params.__dict__)))
        self.p = params
        self.log_dir = os.path.join(out_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.cross_entropy = nn.CrossEntropyLoss()

        tb_dir = os.path.join(out_dir, 'tensorboard')
        self.models_dir = os.path.join(out_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.checkpoint = os.path.join(out_dir, 'checkpoint.pt')
        self.writer = SummaryWriter(tb_dir)
        self.out_json = os.path.join(self.log_dir, 'stat.json')
        self.fixed_test_noise = None
        self.device = device

    def make_shifts(self, latent_dim):
        """
        target indeces contains a batch of random directions (we only have 1 in our case)

        """
        target_indices = torch.randint(
            0, self.p.directions_count, [self.p.batch_size], device=self.device)
        if self.p.shift_distribution == ShiftDistribution.NORMAL:
            # An array of batch_size with random numbers from a a normal distribution.
            shifts = torch.randn(target_indices.shape, device=self.device)
        elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
            shifts = 2.0 * torch.rand(target_indices.shape, device=self.device) - 1.0

        # shifts are scaled
        shifts = self.p.shift_scale * shifts

        # We keep the shifts within some interval.
        shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
        shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        # creates an array of shape ([batch_size, latent_dim])
        z_shift = torch.zeros([self.p.batch_size] + latent_dim, device=self.device)
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            # adds the scaled shifts to z_shifts at the indeces given by target_indeces.
            z_shift[i][index] += val
        # z_shift 
        return target_indices, shifts, z_shift

    def log_train(self, step, should_print=True, stats=()):
        if should_print:
            out_text = '{}% [step {}]'.format(int(100 * step / self.p.n_steps), step)
            for named_value in stats:
                out_text += (' | {}: {:.2f}'.format(*named_value))
            print(out_text)
        for named_value in stats:
            self.writer.add_scalar(named_value[0], named_value[1], step)

        with open(self.out_json, 'w') as out:
            stat_dict = {named_value[0]: named_value[1] for named_value in stats}
            json.dump(stat_dict, out)

    def log_interpolation(self, G, deformator, step):
        noise = make_noise(1, G.dim_z, self.p.truncation).to(self.device)
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
        for z, prefix in zip([noise, self.fixed_test_noise], ['rand', 'fixed']):
            fig = make_interpolation_chart(
                G, deformator, z=z, shifts_r=3 * self.p.shift_scale, shifts_count=3, dims_count=15,
                dpi=500, device=self.device)

            self.writer.add_figure('{}_deformed_interpolation'.format(prefix), fig, step)
            fig_to_image(fig).convert("RGB").save(
                os.path.join(self.images_dir, '{}_{}.jpg'.format(prefix, step)))

    def start_from_checkpoint(self, deformator, shift_predictor):
        step = 0
        if os.path.isfile(self.checkpoint):
            state_dict = torch.load(self.checkpoint)
            step = state_dict['step']
            deformator.load_state_dict(state_dict['deformator'])
            shift_predictor.load_state_dict(state_dict['shift_predictor'])
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, deformator, shift_predictor, step):
        state_dict = {
            'step': step,
            'deformator': deformator.state_dict(),
            'shift_predictor': shift_predictor.state_dict(),
        }
        torch.save(state_dict, self.checkpoint)

    def save_models(self, deformator, shift_predictor, step):
        torch.save(deformator.state_dict(),
                   os.path.join(self.models_dir, 'deformator_{}.pt'.format(step)))
        torch.save(shift_predictor.state_dict(),
                   os.path.join(self.models_dir, 'shift_predictor_{}.pt'.format(step)))

    def log_accuracy(self, G, deformator, shift_predictor, step):
        deformator.eval()
        shift_predictor.eval()

        accuracy = validate_classifier(G, deformator, shift_predictor, trainer=self)
        self.writer.add_scalar('accuracy', accuracy.item(), step)

        deformator.train()
        shift_predictor.train()
        return accuracy

    def log(self, G, deformator, shift_predictor, step, avgs):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, True, [avg.flush() for avg in avgs])

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(G, deformator, step)

        if step % self.p.steps_per_backup == 0 and step > 0:
            self.save_checkpoint(deformator, shift_predictor, step)
            accuracy = self.log_accuracy(G, deformator, shift_predictor, step)
            print('Step {} accuracy: {:.3}'.format(step, accuracy.item()))

        if step % self.p.steps_per_save == 0 and step > 0:
            self.save_models(deformator, shift_predictor, step)

    def log_final(self, G, deformator, shift_predictor, step, avgs):
        self.log_train(step, True, [avg.flush() for avg in avgs])
        self.log_interpolation(G, deformator, step)
        self.save_checkpoint(deformator, shift_predictor, step)
        accuracy = self.log_accuracy(G, deformator, shift_predictor, step)
        print('Step {} accuracy: {:.3}'.format(step, accuracy.item()))
        self.save_models(deformator, shift_predictor, step)

    def train(self, deformator, train_loader, validation_loader, args, multi_gpu=False):
        # G.to(self.device).eval()
        deformator.to(self.device).train()

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) \
            if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None

        avgs = MeanTracker('percent'), MeanTracker('loss'), MeanTracker('direction_loss'),\
               MeanTracker('shift_loss')
        avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss = avgs

        train_losses = []
        for epoch in range(args.epochs):
            # It's approximately 2 seconds for 100 epochs
            total_loss = 0
            for idx, batch in enumerate(train_loader):
                z1, z2 = batch
                
                deformator.zero_grad()

                # Deformation
                shift_prediction = deformator(z1).squeeze(dim = 3)

                z2_prediction = z1 + shift_prediction

                shift_loss = torch.mean(torch.abs(z2_prediction - z2))

                loss = shift_loss
                loss.backward()

                deformator_opt.step()

                avg_loss.add(loss.item())
                total_loss += loss.item()
            total_loss /= len(train_loader)
            train_losses.append(total_loss)
            val_loss = self.validate(deformator, validation_loader, epoch, args)
            val_losses.append(val_loss)

        save_plots(train_losses, val_losses, args)
    
    @torch.no_grad()
    def validate(self, deformator, validation_loader, epoch, args):
        total_loss = 0
        model.eval()
        for idx, batch in enumerate(validation_loader):
            batch = batch.to(args.device)
            z1, z2 = batch
            shift_prediction = deformator(z1).squeeze(dim = 3)
            
            z2_prediction = z1 + shift_prediction

            shift_loss = torch.mean(torch.abs(z2_prediction - z2))

            total_loss += loss.item()
        return total_loss / len(validation_loader)


def save_plots(train_losses, validation_losses, args):
    """Saves loss plots at ../logs/direction/plots
    It includes train_losses and validation losses
    """
    model_name='model_nl'


    PLOTS_PATH = os.path.join('..','logs','direction','plots')
    if not os.path.isdir(PLOTS_PATH):
        os.mkdir(PLOTS_PATH)
    PATH = os.path.join(PLOTS_PATH, args.time_stamp + '.pdf')

    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses, label="training loss")
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    f.savefig(PATH, bbox_inches='tight')

@torch.no_grad()
def validate_classifier(G, deformator, shift_predictor, params_dict=None, trainer=None, device='cpu'):
    n_steps = 100
    if trainer is None:
        trainer = Trainer(params=Params(**params_dict), verbose=False)

    percents = torch.empty([n_steps])
    for step in range(n_steps):
        z = make_noise(trainer.p.batch_size, G.dim_z, trainer.p.truncation).to(trainer.device)
        target_indices, shifts, basis_shift = trainer.make_shifts(deformator.input_dim)

        graphs = G(b_data, z)
        graphs_shifted = G.gen_shifted(b_data, z, deformator(basis_shift))

        logits, _ = shift_predictor(graphs, graphs_shifted)
        percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

    return percents.mean()
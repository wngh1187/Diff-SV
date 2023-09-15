import torch
from torch import nn
import torch.nn.functional as F
from models.modules import GradLogPEstimator2d
from models.base import BaseModule


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise

class Denoiser(BaseModule):
    def __init__(
            self,
            feature_channels,
            inference_steps=10,
    ):
        super().__init__()
        self.diffusion_inference_steps = inference_steps
        self.beta_min = 0.05
        self.beta_max = 20
        self.model = GradLogPEstimator2d(16, n_feats = feature_channels)
      
    def forward(self, x_hat, ref_x=None, is_inference=False):
        with torch.no_grad():
            z = x_hat + torch.randn_like(x_hat, device=x_hat.device) / 1.0
            z0_hat = self.inference(z, x_hat, 10)
        if is_inference: return z0_hat

        ddim_loss = self.calculate_loss(ref_x, x_hat)
        return z0_hat, ddim_loss

    def calculate_loss(self, x0, mu, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True) #
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        noise_estimation = self.model(xt, mu, t)

        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.mean((noise_estimation + z)**2)
        return loss
    
    def inference(self, z, mu, n_timesteps):
        h = 1.0 / n_timesteps
        xt = z

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            dxt = 0.5 * (mu - xt - self.model(xt, mu, t))
            dxt = dxt * noise_t * h
            xt = xt - dxt

        return xt


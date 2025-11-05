import wandb 
from ema_pytorch import EMA
from torch.optim import Adam
from torch.amp import GradScaler
import torch 
import os
from dataloaders.smpl_task_dataset import TaskTrajectoryDataset
from typing import List, Literal
from utils.trainer_utils import cycle
from torch.utils import data
from models.transformers.transformer_object_motion_cond_diffusion import ObjectCondGaussianDiffusion


class DiffusionTrainer():
    def __init__(self, 
        algo_params = None,  
        model_params = None, 
        results_folder=None,
        vis_folder=None,
        root_data_folder = None):

        super().__init__()
        self.algo_params = algo_params
        self.model_params = model_params
        self.root_data_folder = root_data_folder
        assert self.algo_params is not None and self.model_params is not None, f"Either algorithm parmeters or model parameters is none! Got values {self.algo_params} and {self.model_params} respectively!"
        assert results_folder is not None, "Results folder is None! Should be a filepath!"
        assert self.root_data_folder is not None, "Data root folder must not be none!"
        self.use_wandb = self.algo_params.train_params.use_wandb
        if self.use_wandb:
            wandb.init(config=algo_params)
        self.data_in_mem = self.algo_params.train_params.load_ds
        self.train_data_phase = self.algo_params.train_params.data_phase
        assert self.train_data_phase in ("interaction", "navigation"), f"train data phase must be either interaction or navigation, got {self.train_data_phase}"


        self.model = self.build_model()
        self.ema = EMA(self.model, beta=self.algo_params.train_params.ema_decay, update_every=self.algo_params.train_params.ema_update_every)
        self.save_and_sample_every = self.algo_params.train_params.save_and_sample_every
        self.batch_size = self.algo_params.train_params.batch_size
        self.gradient_accumulate_every = self.algo_params.train_params.gradient_accumulate_every
        self.train_num_steps = self.algo_params.train_params.total_timesteps
        self.step = 0  
        self.results_folder = results_folder
        self.device = "cuda" if torch.cuda.is_available else "cpu"

        self.optimizer = Adam(self.model.parameters(), lr=self.algo_params.train_params.learning_rate)
        self.amp = self.algo_params.train_params.amp
        self.scaler = GradScaler(self.device, enabled = self.amp)

        os.makedirs(self.results_folder, exist_ok=True)

        self.viz_folder = os.path.join(self.results_folder, "visualization")
        os.makedirs(self.viz_folder, exist_ok=True)

        self.vis_waypoints = self.algo_params.vis_waypoints

        self.window = self.algo_params.window

        self.add_language_condition = self.algo_params.add_language_condition

        if self.add_language_condition:
            clip_version = "ViT-B/32"
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.train_dataset = TaskTrajectoryDataset(root_data_folder=root_data_folder, phase = self.train_data_phase, data_in_mem=self.data_in_mem)

        print(f"Length of training dataset is {len(self.train_dataset)}")


        self.train_dl = cycle(data.DataLoader(self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=1,))

    def build_model(self):
        self.model_params.architecture_parameters.out_dim = self.get_output_dim()
        if self.train_data_phase == 'interaction':
            self.model = ObjectCondGaussianDiffusion(data_root_folder = self.root_data_folder, **self.model_params.architecture_parameters, **self.model_params.diffusion, **self.model_params.conditioning, **self.model_params.loss_extensions, **self.model_params.sampling)

        else:
            raise NotImplementedError(f"Train data phase {self.train_data_phase} not implemented yet!")

        return self.model
    

    def train(self):
        start_step = self.step
        for idx in range(start_step, self.train_num_steps):
            self.optimizer.zero_grad()
            for i in range(self.gradient_accumulate_every):
                motion_data = next(self.train_dl)
                with torch.amp.autocast(self.device, enabled = self.amp):
                    loss_diffusion = self.model(motion_data, ori_x_cond = None)
                    self.scaler.scale(loss_diffusion/self.gradient_accumulate_every).backward()
                    
                    parameters = [
                        p for p in self.model.parameters() if p.grad is not None
                    ]
                    

                    total_norm = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), 2.0).to(self.device)
                                for p in parameters
                            ]
                        ),
                        2.0,
                    )

                    if torch.isnan(total_norm):
                        print("WARNING: NaN gradients. Skipping to next data...")
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),}

                        wandb.log(log_dict)

                    if idx % 20 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))
                        print("Loss diffusion: %.4f" % (loss_diffusion.item()))
                        if self.add_feet_contact:
                            print("Loss feet: %.4f" % (loss_feet.item()))
                            print("Loss FK: %.4f" % (loss_fk.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

    def save(self, milestone):
        checkpoint = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(
            checkpoint, os.path.join(self.results_folder, "model-" + str(milestone) + ".pt")
        )

    def load(self, milestone):
        checkpoint = torch.load(os.path.join(self.results_folder, "model-" + str(milestone) + ".pt"))
        self.step = checkpoint["step"]
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.ema.load_state_dict(checkpoint["ema"], strict=False)
        self.scaler.load_state_dict(checkpoint["scaler"])

    def get_output_dim(self):
        # Define model
        out_dim = 3 + 9 + 24 * 3 + 22 * 6 + 4# Object relative translation and relative rotation matrix (3 and 9) + smplx model params (the 22 and 24) + use_object_keypoints (4)

        if self.algo_params.add_interaction_feet_contact:
            out_dim += 4

        if self.algo_params.add_interaction_root_xy_ori:
            out_dim += 6

        if self.algo_params.add_wrist_relative:
            out_dim += 18

        return out_dim


def run_diffusion_trainer(cfg):

    DiffusionTrainer(algo_params = cfg.algorithm,  
        model_params = cfg.model,
        results_folder=cfg.algorithm.base_results_folder,
        vis_folder=None,
        root_data_folder = cfg.algorithm.root_data_folder).train()

        


        









        

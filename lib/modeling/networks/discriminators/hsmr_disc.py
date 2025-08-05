from lib.kits.basic import *



class HSMRDiscriminator(nn.Module):

    def __init__(self):
        '''
        Pose + Shape discriminator proposed in HMR
        '''
        super(HSMRDiscriminator, self).__init__()

        self.n_poses = 23
        # poses_alone
        self.D_conv1 = nn.Conv2d(9, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv1.weight)
        nn.init.zeros_(self.D_conv1.bias)
        self.relu = nn.ReLU(inplace=True)
        self.D_conv2 = nn.Conv2d(32, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv2.weight)
        nn.init.zeros_(self.D_conv2.bias)
        pose_out = []
        for i in range(self.n_poses):
            pose_out_temp = nn.Linear(32, 1)
            nn.init.xavier_uniform_(pose_out_temp.weight)
            nn.init.zeros_(pose_out_temp.bias)
            pose_out.append(pose_out_temp)
        self.pose_out = nn.ModuleList(pose_out)

        # betas
        self.betas_fc1 = nn.Linear(10, 10)
        nn.init.xavier_uniform_(self.betas_fc1.weight)
        nn.init.zeros_(self.betas_fc1.bias)
        self.betas_fc2 = nn.Linear(10, 5)
        nn.init.xavier_uniform_(self.betas_fc2.weight)
        nn.init.zeros_(self.betas_fc2.bias)
        self.betas_out = nn.Linear(5, 1)
        nn.init.xavier_uniform_(self.betas_out.weight)
        nn.init.zeros_(self.betas_out.bias)

        # poses_joint
        self.D_alljoints_fc1 = nn.Linear(32*self.n_poses, 1024)
        nn.init.xavier_uniform_(self.D_alljoints_fc1.weight)
        nn.init.zeros_(self.D_alljoints_fc1.bias)
        self.D_alljoints_fc2 = nn.Linear(1024, 1024)
        nn.init.xavier_uniform_(self.D_alljoints_fc2.weight)
        nn.init.zeros_(self.D_alljoints_fc2.bias)
        self.D_alljoints_out = nn.Linear(1024, 1)
        nn.init.xavier_uniform_(self.D_alljoints_out.weight)
        nn.init.zeros_(self.D_alljoints_out.bias)


    def forward(self, poses_body: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the discriminator.
        ### Args
        - poses: torch.Tensor, shape (B, 23, 9) 
            - Matrix representation of the SKEL poses excluding the global orientation.
        - betas: torch.Tensor, shape (B, 10)
        ### Returns
        - torch.Tensor, shape (B, 25)
        '''
        poses_body = poses_body.reshape(-1, self.n_poses, 1, 9)  # (B, n_poses, 1, 9)
        B = poses_body.shape[0]
        poses_body = poses_body.permute(0, 3, 1, 2).contiguous()  # (B, 9, n_poses, 1)

        # poses_alone
        poses_body = self.D_conv1(poses_body)
        poses_body = self.relu(poses_body)
        poses_body = self.D_conv2(poses_body)
        poses_body = self.relu(poses_body)

        poses_out = []
        for i in range(self.n_poses):
            poses_out_i = self.pose_out[i](poses_body[:, :, i, 0])
            poses_out.append(poses_out_i)
        poses_out = torch.cat(poses_out, dim=1)

        # betas
        betas = self.betas_fc1(betas)
        betas = self.relu(betas)
        betas = self.betas_fc2(betas)
        betas = self.relu(betas)
        betas_out = self.betas_out(betas)

        # poses_joint
        poses_body = poses_body.reshape(B, -1)
        poses_all = self.D_alljoints_fc1(poses_body)
        poses_all = self.relu(poses_all)
        poses_all = self.D_alljoints_fc2(poses_all)
        poses_all = self.relu(poses_all)
        poses_all_out = self.D_alljoints_out(poses_all)

        disc_out = torch.cat((poses_out, betas_out, poses_all_out), dim=1)
        return disc_out
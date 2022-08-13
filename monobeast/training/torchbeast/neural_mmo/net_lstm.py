import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbeast.core.mask import MaskedPolicy


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class NMMONet(nn.Module):

    def __init__(self, observation_space, num_actions, use_lstm=False):
        super().__init__()
        #new
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2), nn.ReLU(), #8
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU()) #8


        self.global_cnn = nn.Sequential(
            nn.Conv2d(in_channels=9,
                      out_channels=32,
                      kernel_size=15,
                      stride=5,
                      padding=0), nn.ReLU(), #30
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=0), nn.ReLU(), #14
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=2), nn.ReLU()) #8

        # self.core = nn.Linear(32 * 15 * 15, 512)
        # self.policy = nn.Linear(512, num_actions)
        # self.baseline = nn.Linear(512, 1)
        
        self.local_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.global_fc1 = nn.Linear(64 * 8 * 8, 256)    

        self.player_info_fc1 =  nn.Linear(26, 128) 
        self.player_info_fc2 =  nn.Linear(128, 128)    

        self.cur_task_fc1 =  nn.Linear(5, 64) 
        self.cur_task_fc2 =  nn.Linear(64, 64)  

        self.qkv_proj = nn.Linear(384, 3*384)
        self.attention = nn.MultiheadAttention(384, 4, batch_first=True)

        self.gru = nn.GRU(384*2 + 256 + 64, hidden_size=512, num_layers=1, batch_first=True)

        self.final_fc = nn.Linear(512, 128)

        self.policy = nn.Linear(128, num_actions)
        
        self.baseline = nn.Linear(128, 1)

        #https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/4.PPO-discrete/ppo_discrete.py
        # if args.use_orthogonal_init:
        #     print("------use_orthogonal_init------")
        #     orthogonal_init(self.fc1)
        #     orthogonal_init(self.fc2)
        #     orthogonal_init(self.fc3, gain=0.01)


    def initial_state(self, batch_size=1):
        return tuple()


    def forward(self, input_dict, h = None, training=False):
        # [T, B, ...]

        terrain = input_dict["terrain"]                             #[1, 8, 15, 15]
        camp = input_dict["camp"]                                   #[1, 8, 15, 15]
        entity = input_dict["entity"]                               #[1, 8, 7, 15, 15]
        player_info = input_dict["player_info"]                     #[1, 8, 26]
        cur_task = input_dict["cur_task"][:,0,:].reshape(-1, 5)     #[1, 5]
        global_terrain = input_dict["global_terrain"][:,0,:,:].reshape(-1, 160, 160)               #[1, 160, 160]
        global_camp = input_dict["global_camp"][:,0,:,:].reshape(-1, 160, 160)                     #[1, 160, 160]

        terrain = F.one_hot(terrain, num_classes=6).permute(0, 1, 4, 2, 3)                          #[1, 8, 6, 15, 15]
        camp = F.one_hot(camp, num_classes=4).permute(0, 1, 4, 2, 3)[:,:,1:4,:,:]                   #[1, 8, 3, 15, 15]
        global_terrain = F.one_hot(global_terrain, num_classes=6).permute(0, 3, 1, 2)            #[1, 6, 160, 160]
        global_camp = F.one_hot(global_camp, num_classes=4).permute(0, 3, 1, 2)[:,1:4,:,:]     #[1, 3, 160, 160]

        x = torch.cat([terrain, camp, entity], dim=2)                                               #[1, 8, 16, 15, 15]
        global_x = torch.cat([global_terrain, global_camp], dim=1).to(torch.float)                                  #[1, 9, 15, 15]

        # print(terrain.shape)
        # print(camp.shape)
        # exit()
        #print(x.shape)
        T, B, C, H, W = x.shape
        # assert C == 17 and H == W == 15
        x = torch.flatten(x, 0, 1)                                                                  #[TXB, 16, 15, 15]
        player_info_x = torch.flatten(player_info, 0, 1)

        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.local_fc1(x))

        #print("global_x", global_x.dtype, global_x.shape)
        global_x = self.global_cnn(global_x)
        global_x = torch.flatten(global_x, start_dim=1)
        global_x = F.relu(self.global_fc1(global_x))

        player_info_x = F.relu(self.player_info_fc1(player_info_x))
        player_info_x = F.relu(self.player_info_fc2(player_info_x))

        cur_task_x = F.relu(self.cur_task_fc1(cur_task))
        cur_task_x = F.relu(self.cur_task_fc2(cur_task_x))


        x = torch.cat([x, player_info_x], dim=-1).reshape(T, B, -1)                 #[T, B, 384]

        #x attention
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        atten_output, attn_output_weights = self.attention(q, k ,v)  #[T, B, 384]

        global_x =  torch.cat([global_x, cur_task_x], dim=1).unsqueeze(1).repeat(1, B, 1)    #[T, B, 320]

        #print(x.shape, atten_output.shape, global_x.shape)
        x = torch.cat([x, atten_output, global_x], dim=-1)      #[T, B, 384 + 384 + 320]

        x, h = self.gru(x, h)
        x = self.final_fc(x)

        logits = self.policy(x)
        baseline = self.baseline(x)

        va = input_dict.get("va", None)
        if va is not None:
            va = torch.flatten(va, 0, 1)

        dist = MaskedPolicy(logits, valid_actions=va)
        if not training:
            action = dist.sample()
            action = action.view(T, B)
        else:
            action = None
        policy_logits = dist.logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        output = dict(policy_logits=policy_logits, baseline=baseline)
        if action is not None:
            output["action"] = action
        return (output, h)
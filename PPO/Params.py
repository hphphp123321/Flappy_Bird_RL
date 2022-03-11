import argparse

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
# args for device
parser.add_argument('--device', type=str, default="cpu", help='Number of jobs of instances')
# args for env
parser.add_argument('--np_seed_train', type=int, default=200, help='Seed for numpy for training')
parser.add_argument('--np_seed_validation', type=int, default=200, help='Seed for numpy for validation')
parser.add_argument('--torch_seed', type=int, default=600, help='Seed for torch')
# args for network
parser.add_argument('--input_dim_actor', type=int, default=8, help='number of dimension of state number')
parser.add_argument('--input_dim_critic', type=int, default=8, help='number of dimension of state number')
parser.add_argument('--output_dim_actor', type=int, default=2, help='number of dimension of action number')
parser.add_argument('--output_dim_critic', type=int, default=1, help='number of dimension of v(s)')

parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='number of layers in actor MLP')
parser.add_argument('--hidden_dim_actor', type=int, default=256, help='hidden dim of MLP in actor')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='number of layers in critic MLP')
parser.add_argument('--hidden_dim_critic', type=int, default=256, help='hidden dim of MLP in critic')
# args for PPO
# parser.add_argument('--num_envs', type=int, default=4, help='No. of envs for training')
parser.add_argument('--max_episodes', type=int, default=500000, help='max of episodes')
parser.add_argument('--batch_size', type=int, default=512, help='batch from memories')
parser.add_argument('--lr', type=float, default=3e-4, help='lr')
parser.add_argument('--decay_flag', type=bool, default=True, help='lr decayflag')
parser.add_argument('--decay_step_size', type=int, default=5000, help='decay_step_size')
parser.add_argument('--decay_ratio', type=float, default=0.9, help='decay_ratio, e.g. 0.9, 0.95')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--k_epochs', type=int, default=16, help='update policy for K epochs')
parser.add_argument('--memory_maxlen', type=int, default=5000, help='max size of memory')
parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
parser.add_argument('--vloss_coef', type=float, default=1, help='critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=2, help='policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='entropy loss coefficient')

configs = parser.parse_args()

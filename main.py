import os
import random
import time
import wandb
import torch
from datasets.Dataset import Dataset
from datasets.InD import InD
from datasets.EthUcy import EthUcy
from datasets.AV2_parallel import AV2, AV2Dataset, AV2ObservationSite
from model.TrajFlow import TrajFlow, CausalEnocder, Flow
from train import train
from evaluate import evaluate
from visualize import visualize
from visualize_temp import visualize_temp
from visualize_av2 import visualize_av2
from visualize_av2_minimal import export_for_notebook


should_train = False
should_serialize = True
should_evaluate = True
should_visualize = False
simple_visualization = False
verbose = False
marginal = True

# AV2-specific switches
# If True, main.py will NOT rebuild AV2 cache; it will load from an existing
# .pt file and construct DataLoaders directly. You can change av2_cache_path.
use_av2_cache_only = True
av2_cache_path = "data/av2_mf_tiny/av2_cache_2000.pt"
# with wandb.init() as run:
with wandb.init(mode="offline") as run:
	run.config.setdefaults({
		'seed': random.randint(0, 2**32 - 1),
		'encoder': 'CDE',
		'flow': 'CNF',
		'dataset': 'AV2',
		'observation_site': 'zara2',
		'masked_data_ratio': 0
	})

	torch.manual_seed(run.config.seed)

	causal_encoder=CausalEnocder[run.config.encoder]
	flow=Flow[run.config.flow]
	dataset = Dataset[run.config.dataset]

	seq_len = 0
	input_dim = 0
	feature_dim = 0
	embedding_dim = 0
	hidden_dim = 0

	observation_site = None

	if dataset == Dataset.InD:
		seq_len = 100
		input_dim = 2
		feature_dim = 5
		embedding_dim = 128
		hidden_dim = 512
		training_epochs = 25
		evaulation_samples = 1000
		norm_rotate = False

		ind = InD(
			root="data",
			train_ratio=0.75, 
			train_batch_size=64, 
			test_batch_size=1,
			missing_rate=run.config.masked_data_ratio)
		observation_site = ind.observation_site1
	elif dataset == Dataset.EthUcy:
		seq_len = 12
		input_dim = 2
		feature_dim = 4
		embedding_dim = 128
		hidden_dim = 512
		training_epochs = 150
		evaulation_samples = 20
		norm_rotate = True

		ethucy = EthUcy(train_batch_size=128, test_batch_size=1, history=8, futures=12, smin=0.3, smax=1.7)
		observation_site = (
        	ethucy.eth_observation_site if run.config.observation_site == 'eth' else
        	ethucy.hotel_observation_site if run.config.observation_site == 'hotel' else
        	ethucy.univ_observation_site if run.config.observation_site == 'univ' else
			ethucy.zara1_observation_site if run.config.observation_site == 'zara1' else
			ethucy.zara2_observation_site if run.config.observation_site == 'zara2' else
        	ethucy.zara2_observation_site
    	)
	elif dataset == Dataset.AV2:
		seq_len = 50
		input_dim = 2
		# AV2 features include an extra time channel appended in datasets/AV2.py
		feature_dim = 6
		embedding_dim = 128
		hidden_dim = 512
		train_ratio = 0.8
		training_epochs = 40
		evaulation_samples = 100
		norm_rotate = False

		if use_av2_cache_only:
			# ---- Load precomputed AV2 cache from a .pt file ----
			# Expected format (as saved by datasets/AV2_parallel.py):
			# {
			#   "positions": (N, TOTAL_STEPS, 2) normalized to [0,1],
			#   "features":  (N, HISTORY_STEPS, 6) with time channel,
			#   "spatial_boundaries": (2, 2)
			# }
			print(f"Loading AV2 cache from: {av2_cache_path}")
			cache = torch.load(av2_cache_path, map_location="cpu", weights_only=False)
			positions = cache["positions"]
			features = cache["features"]
			spatial_boundaries = cache["spatial_boundaries"]

			if not torch.is_tensor(positions):
				positions = torch.as_tensor(positions)
			if not torch.is_tensor(features):
				features = torch.as_tensor(features)
			if not torch.is_tensor(spatial_boundaries):
				spatial_boundaries = torch.as_tensor(spatial_boundaries)

			positions = positions.float()
			features = features.float()
			spatial_boundaries = spatial_boundaries.float()

			# Ensure time channel is present (compatibility with older caches)
			if features.ndim == 3 and features.shape[-1] == 5:
				N = features.shape[0]
				from datasets.AV2 import HISTORY_STEPS  # keep import local to avoid circular issues
				t = torch.linspace(0.0, 2.0, HISTORY_STEPS).unsqueeze(0).unsqueeze(-1)
				t = t.expand(N, HISTORY_STEPS, 1)
				features = torch.cat([features, t], dim=-1)

			dataset = AV2Dataset(positions, features)
			train_size = int(len(dataset) * train_ratio)
			test_size = len(dataset) - train_size
			train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

			train_loader = torch.utils.data.DataLoader(
				train_dataset, batch_size=64, shuffle=True
			)
			test_loader = torch.utils.data.DataLoader(
				test_dataset, batch_size=1, shuffle=False
			)

			observation_site = AV2ObservationSite(spatial_boundaries, train_loader, test_loader)
			print(len(observation_site.train_loader.dataset))
			print(len(observation_site.test_loader.dataset))
			
			av2_map_root = "data/av2_mf_tiny"
		else:
			# Default path: construct AV2 object which will build or reuse cache internally.
			av2 = AV2(
				root="data/av2_mf_tiny",
				train_ratio=train_ratio,
				train_batch_size=64,
				test_batch_size=1,
				max_scenarios=None)
			observation_site = av2.observation_site
			print(len(observation_site.train_loader.dataset))
			print(len(observation_site.test_loader.dataset))
			
			av2_map_root = av2.root
	else:
		raise ValueError(f'{dataset.name} is not an experiment dataset')

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	traj_flow = TrajFlow(
		seq_len=seq_len, 
		input_dim=input_dim, 
		feature_dim=feature_dim, 
		embedding_dim=embedding_dim,
		hidden_dim=hidden_dim,
		causal_encoder=causal_encoder,
		flow=flow,
		marginal=marginal,
		norm_rotation=norm_rotate).to(device)
	
	num_parameters = sum(p.numel() for p in traj_flow.parameters() if p.requires_grad)
	if verbose:
		print(f'parameters: {num_parameters}')
	wandb.log({'parameters': num_parameters})

	train_start_time = time.time()

	total_loss = []
	if should_train:
		total_loss = train(
			observation_site=observation_site,
			model=traj_flow,
			epochs=training_epochs,
			lr=1e-3,
			weight_decay=0,
			gamma=0.999,
			verbose=verbose,
			device=device)
		
	train_end_time = time.time()
	train_runtime = train_end_time - train_start_time
	if verbose:
		print(train_runtime)
	wandb.log({'train runtime': train_runtime})

	traj_flow.eval()
	input, feature, _ = next(iter(observation_site.test_loader))
	input = input.to(device)
	feature = feature.to(device)
	inference_start_time = time.time()
	traj_flow.sample(input, feature, 100, 100)
	inference_end_time = time.time()
	inference_runtime = inference_end_time - inference_start_time
	if verbose:
		print(inference_runtime)
	wandb.log({'inference runtime': inference_runtime})
		
	for loss in total_loss:
		wandb.log({'loss': loss})
			
	if should_serialize:
		#suffix = 'marginal' if marginal else 'joint'
		#model_name = f'trajflow_{suffix}_{run.config.dataset}_{run.config.seed}.pt'
		suffix = 'marginal' if marginal else 'joint'
		model_name = f"trajflow_{run.config.encoder}_{run.config.flow}_{suffix}_{run.config.dataset}.pt"
		if should_train:
			torch.save(traj_flow.state_dict(), model_name)
		elif os.path.exists(model_name):
			state = torch.load(model_name, map_location=device, weights_only=False)
			model_state = traj_flow.state_dict()
			compatible = {k: v for k, v in state.items() if k in model_state and getattr(v, "shape", None) == model_state[k].shape}
			missing, unexpected = traj_flow.load_state_dict(compatible, strict=False)
			if verbose:
				print(f"loaded compatible keys: {len(compatible)}; missing: {len(missing)}; unexpected: {len(unexpected)}")

	if should_evaluate:
		rmse, crps, min_ade, min_fde, nll = evaluate(
			observation_site=observation_site,
			model=traj_flow,
			num_samples=evaulation_samples,
			device=device)
		
		if verbose:
			print(f'rmse: {rmse}')
			print(f'crps: {crps}')
			print(f'min ade: {min_ade}')
			print(f'min fde: {min_fde}')
			print(f'nll: {nll}')
		wandb.log({'rmse': rmse, 'crps': crps, 'min ade': min_ade, 'min fde': min_fde, 'nll': nll})

	if should_visualize:
		if dataset == Dataset.AV2:
			# visualize_av2(
			# 	observation_site=observation_site,
			# 	model=traj_flow,
			# 	map_root=av2_map_root,
			# 	num_samples=1, # 5
	 		# 	steps=50, # 200
			# 	prob_threshold=0.001,
			# 	output_dir='visualization',
			# 	simple=simple_visualization,
			# 	device=device)
			export_for_notebook(
				observation_site=observation_site,
				model=traj_flow,
				num_samples=1,
				steps=50,              
				output_dir="visualization",
				device=device,
			)
		else:
			visualize(
				observation_site=observation_site,
				model=traj_flow,
				num_samples=30,
				steps=1000,
				prob_threshold=0.001,
				output_dir='visualization',
				simple=simple_visualization,
				device=device)
		# visualize_temp(
		# 	data_loader=observation_site.test_loader,
		# 	model=traj_flow,
		# 	num_samples=2,
		# 	steps=100,
		# 	prob_threshold=0.001,
		# 	output_dir='visualization_temp',
		# 	simple=simple_visualization,
		# 	device=device)

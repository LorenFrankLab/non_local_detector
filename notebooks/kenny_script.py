import numpy as np
from track_linearization import make_track_graph

from non_local_detector import (
    ClusterlessDecoder,
    ContFragClusterlessClassifier,
    Environment,
)

## Parameters
sampling_frequency = 500  # samples/s
bin_size = 2.0  # cm
clusterless_algorithm = "clusterless_kde"
clusterless_algorithm_params = {
    "position_std": 3.5,  # cm, controls the width of the kernel for KDE over position
    "waveform_std": 24.0,  # uV, controls the width of the kernel for KDE over waveform features
    "block_size": int(2**10),  # number of spikes to process at once
}

# Create a 2D environment for decoding
env_2D = Environment(place_bin_size=bin_size)

## Decoder1: Continuous-Fragmented (Random walk and jump transitions), 2D environment
# Fit the classifier to position and spike data
classifier = ContFragClusterlessClassifier(
    environments=env_2D,
    clusterless_algorithm=clusterless_algorithm,
    clusterless_algorithm_params=clusterless_algorithm_params,
    sampling_frequency=sampling_frequency,
).fit(
    position_time=position_info.index.to_numpy(),
    position=position_info.loc[:, ["head_position_x", "head_position_y"]].to_numpy(),
    spike_times=spike_times,
    spike_waveform_features=spike_waveform_features,
)
# Predict posterior for a subset of time points
classifier_results = classifier.predict(
    time=position_info.index.to_numpy()[100:200],
    spike_times=spike_times,
    spike_waveform_features=spike_waveform_features,
)

# Get the acausal posterior summed over state dimension
posterior = classifier_results.acausal_posterior.unstack("state_bins").sum("state")

## Decoder2: Only random walk transition, 2D environment
# Fit the standard clusterless decoder
decoder = ClusterlessDecoder(
    environments=env_2D,
    clusterless_algorithm=clusterless_algorithm,
    clusterless_algorithm_params=clusterless_algorithm_params,
).fit(
    position_time=position_info.index.to_numpy(),
    position=position_info.loc[:, ["head_position_x", "head_position_y"]].to_numpy(),
    spike_times=spike_times,
    spike_waveform_features=spike_waveform_features,
)

# Predict posterior for a subset of time points
decoder_results = decoder.predict(
    time=position_info.index.to_numpy()[100:200],
    spike_times=spike_times,
    spike_waveform_features=spike_waveform_features,
)
# Get the acausal posterior
posterior2 = decoder_results.acausal_posterior.unstack("state_bins")

## Decoder3: 1D environment with a track graph
# Define the track graph structure (edges and node positions)
edge_order = [(9, 6), (6, 0), (6, 1), (9, 7), (7, 2), (7, 3), (9, 8), (8, 4), (8, 5)]
edge_spacing = 15
node_positions = np.array(
    [
        [179.4527997, 45.06356539],
        [225.71560744, 123.25220539],
        [184.17870421, 205.02047636],
        [88.93641002, 209.70195249],
        [37.85507841, 128.45589571],
        [88.19751712, 45.06356539],
        [179.63427582, 99.71153958],
        [135.43811066, 180.45759636],
        [87.45862421, 99.55227765],
        [135.5973726, 127.33183636],
    ]
)
# Create the track graph object
track_graph = make_track_graph(
    node_positions=node_positions,
    edges=edge_order,
)

# Create a 1D environment using the track graph
env_1D = Environment(
    place_bin_size=bin_size,
    track_graph=track_graph,
    edge_spacing=edge_spacing,
    edge_order=edge_order,
)

# Fit the decoder to position and spike data in 1D environment
decoder = ClusterlessDecoder(
    environments=env_1D,
    clusterless_algorithm=clusterless_algorithm,
    clusterless_algorithm_params=clusterless_algorithm_params,
).fit(
    position_time=position_info.index.to_numpy(),
    position=position_info.loc[:, ["head_position_x", "head_position_y"]].to_numpy(),
    spike_times=spike_times,
    spike_waveform_features=spike_waveform_features,
)

# Predict posterior for a subset of time points in 1D environment
decoder_results_1D = decoder.predict(
    time=position_info.index.to_numpy()[100:200],
    spike_times=spike_times,
    spike_waveform_features=spike_waveform_features,
)
# Get the acausal posterior
posterior3 = decoder_results_1D.acausal_posterior.unstack("state_bins")

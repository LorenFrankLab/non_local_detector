# Notebooks Organization

This directory contains Jupyter notebooks for testing, developing, and demonstrating the `non_local_detector` package. Notebooks are organized into thematic categories for easy navigation.

## Directory Structure

### 01_models_and_validation/

Core decoder models, testing, and validation notebooks. These demonstrate the primary functionality of the package.

| Notebook | Description |
|----------|-------------|
| `sorted_spikes_detector_test.ipynb` | Tests `NonLocalSortedSpikesDetector` with simulated data, Viterbi algorithm implementation |
| `decoder_simulation_test.ipynb` | Comprehensive simulation testing with `ContFragSortedSpikesClassifier` and `SortedSpikesDecoder` |
| `decoder_2D_real_data.ipynb` | 2D position decoding on real dataset (Jaq_03_16) |
| `posterior_consistency_validation.ipynb` | Model validation using posterior consistency metrics (KL divergence, HPD overlap) |
| `model_checking_real_data.ipynb` | Real data analysis and posterior computation with GPU support |
| `chunked_filtering_test.ipynb` | Tests forward-backward algorithm with chunked data processing |
| `nospike_cont_frag_states.ipynb` | Custom detector with No-Spike, Continuous, and Fragmented states |
| `sorted_spikes_test.ipynb` | General sorted spikes decoder testing |

### 02_likelihood_models/

Likelihood model implementations, place field estimation, and optimization algorithms.

| Notebook | Description |
|----------|-------------|
| `kde_implementation_comparison.ipynb` | Comparison of KDE implementations (sklearn vs JAX) for 1D and 2D cases |
| `weighted_kde_test.ipynb` | Weighted kernel density estimation testing |
| `weighted_place_fields.ipynb` | Weighted place field computation methods |
| `place_field_fitting_synthetic.ipynb` | Place field fitting on synthetic data |
| `place_field_fitting_real_data_v1.ipynb` | Place field fitting on real data (version 1) |
| `place_field_fitting_real_data_v2.ipynb` | Place field fitting on real data (version 2) |
| `likelihood_EM_algorithm.ipynb` | Full expectation-maximization algorithm for likelihood estimation |
| `gradient_descent_optimization.ipynb` | Gradient descent for linear regression, Poisson regression, and KDE regression |

### 03_state_transitions/

State transition models and spike-timing based decoding approaches.

| Notebook | Description |
|----------|-------------|
| `nonstationary_discrete_transitions.ipynb` | Non-stationary discrete state transition matrix testing |
| `spike_time_decoding_v1.ipynb` | Spike time-based decoding approach (version 1) |
| `spike_time_decoding_v2.ipynb` | Spike time-based decoding approach (version 2) |
| `spike_times_test.ipynb` | General spike timing tests |

### 04_visualization/

Interactive visualization tools and data exploration notebooks.

| Notebook | Description |
|----------|-------------|
| `interactive_2D_decoder_playback.ipynb` | HoloViews-based interactive 2D decoder visualization with playback controls |
| `interactive_streaming_optimized.ipynb` | Optimized streaming visualization with Panel/HoloViews |
| `data_loading_exploration.ipynb` | Data loading utilities and initial exploration |

### 05_experimental/

Experimental and advanced methods not part of the core package.

| Notebook | Description |
|----------|-------------|
| `normalizing_flows_marked_process.ipynb` | Normalizing flows (Flax/distrax) for marked point processes |
| `normalizing_flows_basics.ipynb` | Introduction to normalizing flows theory and implementation |
| `sensor_fusion_ukf_rts.ipynb` | Augmented Unscented Kalman Filter with RTS smoother for rat tracking with IMU/camera fusion |
| `recurrent_marked_temporal_point_process.ipynb` | Recurrent Marked Temporal Point Process (RMTPP) model with encoder-decoder architecture |
| `test_modifications.ipynb` | Miscellaneous testing and modifications |

## Usage Guidelines

### Running Notebooks

1. **Activate the environment**:
   ```bash
   conda activate non_local_detector
   ```

2. **Start Jupyter**:
   ```bash
   jupyter lab
   ```

3. **Navigate** to the appropriate category folder and open the notebook of interest.

### Adding New Notebooks

When creating new notebooks, please:

1. Choose the appropriate category folder based on the notebook's primary purpose
2. Use descriptive, snake_case filenames that indicate the notebook's content
3. Include a clear title and description in the first markdown cell
4. Document any specific data requirements or dependencies

### Notebook Naming Convention

- Use descriptive names that reflect content: `feature_name_test.ipynb`
- For multiple versions, append version number: `feature_v1.ipynb`, `feature_v2.ipynb`
- Avoid generic prefixes like "test_" unless specifically testing functionality
- Use snake_case for all filenames

## Data Requirements

Many notebooks require access to neural recording data. Common datasets referenced:

- **Jaq_03_16**: Real hippocampal recording dataset used in multiple validation notebooks
- **Simulated data**: Generated within notebooks for testing purposes

## Related Documentation

- [Project README](../README.md) - Package overview and installation
- [CLAUDE.md](../CLAUDE.md) - Development guidelines and architecture
- [Source code](../src/non_local_detector/) - Package implementation

## Notes

- Notebooks in `05_experimental/` represent exploratory work and may not be actively maintained
- For production use cases, refer to the package API documentation rather than notebooks
- Some notebooks may require GPU support (JAX with CUDA) for optimal performance

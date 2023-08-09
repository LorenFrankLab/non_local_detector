try:
    from non_local_detector.visualization.figurl_1D import (
        create_interactive_1D_decoding_figurl,
    )
    from non_local_detector.visualization.figurl_2D import (
        create_interactive_2D_decoding_figurl,
    )
except ImportError:
    pass
from non_local_detector.visualization.static import plot_non_local_model

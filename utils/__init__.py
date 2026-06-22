# utils/__init__.py
# Makes utils a proper Python package so you can do:
#   from utils.common import set_seed, get_device
#   from utils.data import make_classification_data, build_vocab
#   from utils.viz import plot_loss_curve, plot_attention_heatmap

from .common import set_seed, get_device, Timer
from .data import (
    make_classification_data,
    make_sequence_data,
    make_regression_data,
    simple_tokenize,
    build_vocab,
    encode,
    pad_sequences,
    train_test_split,
)
from .viz import (
    plot_loss_curve,
    plot_attention_heatmap,
    plot_embeddings_2d,
    plot_decision_boundary,
)
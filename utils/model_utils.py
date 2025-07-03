from models import FullyConnectedModel

def build_model(input_size, num_classes, layer_sizes, 
                use_batchnorm=False, use_dropout=False, dropout_rate=0.5):
    layers = []
    prev_size = input_size

    for size in layer_sizes:
        layers.append({"type": "linear", "size": size})
        if use_batchnorm:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if use_dropout:
            layers.append({"type": "dropout", "rate": dropout_rate})
        prev_size = size

    return FullyConnectedModel(
        input_size=input_size,
        num_classes=num_classes,
        layers=layers
    )

def generate_architectures(depth_list, width_list):
    """
    Возвращает список конфигураций: (depth, width, layer_sizes)
    """
    configs = []
    for depth in depth_list:
        for width in width_list:
            layer_sizes = [width] * depth
            configs.append((depth, width, layer_sizes))
            
    return configs
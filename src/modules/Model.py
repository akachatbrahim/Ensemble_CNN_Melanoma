import tensorflow as tf
from src.config.Config import learning_rate, weight_decay,input_shape,num_classes
from tensorflow.keras import models, layers

# ============================================================================
# AVAILABLE DL MODELS CONFIGURATION (11 ARCHITECTURES)
# ============================================================================
DL_MODELS = {
    "1": {
        "name": "ResNet50",
        "module": "resnet50",
        "model_class": "ResNet50",
        "preprocess": "resnet50",
        "description": "ResNet50 - 50-layer with residual connections (balanced)",
        "type": "keras"
    },
    "2": {
        "name": "MobileNetV2",
        "module": "mobilenet_v2",
        "model_class": "MobileNetV2",
        "preprocess": "mobilenet_v2",
        "description": "MobileNetV2 - Lightweight, mobile-optimized (fast)",
        "type": "keras"
    },
    "3": {
        "name": "EfficientNetV2B0",
        "module": "efficientnet_v2",
        "model_class": "EfficientNetV2B0",
        "preprocess": "efficientnet_v2",
        "description": "EfficientNetV2B0 - Efficient scaling (faster, better)",
        "type": "keras"
    },
    "4": {
        "name": "EfficientNetV2B2",
        "module": "efficientnet_v2",
        "model_class": "EfficientNetV2B2",
        "preprocess": "efficientnet_v2",
        "description": "EfficientNetV2B2 - Efficient scaling (larger, more accurate)",
        "type": "keras"
    },
    "5": {
        "name": "EfficientNetV2B3",
        "module": "efficientnet_v2",
        "model_class": "EfficientNetV2B3",
        "preprocess": "efficientnet_v2",
        "description": "EfficientNetV2B3 - Efficient scaling (larger variant)",
        "type": "keras"
    },
    "6": {
        "name": "EfficientNetV2M",
        "module": "efficientnet_v2",
        "model_class": "EfficientNetV2M",
        "preprocess": "efficientnet_v2",
        "description": "EfficientNetV2M - Efficient scaling (medium, high accuracy)",
        "type": "keras"
    },
    "7": {
        "name": "DenseNet121",
        "module": "densenet",
        "model_class": "DenseNet121",
        "preprocess": "densenet",
        "description": "DenseNet121 - Dense connections (balanced)",
        "type": "keras"
    },
    "8": {
        "name": "DenseNet169",
        "module": "densenet",
        "model_class": "DenseNet169",
        "preprocess": "densenet",
        "description": "DenseNet169 - Dense connections (high accuracy)",
        "type": "keras"
    },
    "9": {
        "name": "VGG16",
        "module": "vgg16",
        "model_class": "VGG16",
        "preprocess": "vgg16",
        "description": "VGG16 - 16-layer CNN (classic)",
        "type": "keras"
    },
    "10": {
        "name": "VGG19",
        "module": "vgg19",
        "model_class": "VGG19",
        "preprocess": "vgg19",
        "description": "VGG19 - 19-layer CNN (classic, accurate)",
        "type": "keras"
    },
    "11": {
        "name": "CNN",
        "description": "CNN - Custom convolutional neural network",
        "type": "custom"
    },
    "12": {
        "name": "ConvNeXtSmall",
        "type": "keras",
        "module": "convnext",
        "model_class": "ConvNeXtSmall",
        "preprocess": "convnext",
        "description": "Modern CNN ConvNeXt Small"
    }
}

def load_model_and_preprocess(model_info):
    """
    Load a model class/function and its preprocessing function based on model info.
    Returns: (Model_class_or_fn, preprocess_fn) - tuple of (callable that creates model, function)
    Handles special cases for VGG16, VGG19, CNN, and ConvNeXtSmall.
    """
    import tensorflow.keras.applications as tfa
    
    model_type = model_info.get('type')
    model_name = model_info.get('name')
    
    # Handle custom CNN model
    if model_type == 'custom':
        return (build_model_cnn, None)  # Return the function itself, not the result
    
    # Handle keras applications
    try:
        module_name = model_info['module']
        model_class_name = model_info['model_class']
        preprocess_name = model_info['preprocess']
        
        # Special handling for VGG models (they are directly in tfa)
        if module_name in ['vgg16', 'vgg19']:
            try:
                # Try direct import from tfa (in newer TensorFlow versions)
                model_module = getattr(tfa, module_name)
                Model = getattr(model_module, model_class_name)
                preprocess_module = getattr(tfa, preprocess_name)
                preprocess_fn = getattr(preprocess_module, 'preprocess_input')
                
                # Validate: Model should be callable (it's a function or class)
                if not callable(Model):
                    raise TypeError(f"Expected {model_class_name} to be callable, got {type(Model)}")
                if not callable(preprocess_fn):
                    raise TypeError(f"Expected preprocess_input to be callable, got {type(preprocess_fn)}")
                
                return Model, preprocess_fn
            except (AttributeError, TypeError) as e:
                # Fallback for older TensorFlow versions
                if module_name == 'vgg16':
                    from tensorflow.keras.applications.vgg16 import VGG16 as Model
                    from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_fn
                elif module_name == 'vgg19':
                    from tensorflow.keras.applications.vgg19 import VGG19 as Model
                    from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_fn
                else:
                    raise
                
                if not callable(Model):
                    raise TypeError(f"Expected {model_class_name} to be callable, got {type(Model)}")
                if not callable(preprocess_fn):
                    raise TypeError(f"Expected preprocess_input to be callable, got {type(preprocess_fn)}")
                    
                return Model, preprocess_fn
        
        # Special handling for ConvNeXt (might not be available in all TF versions)
        elif module_name == 'convnext':
            try:
                model_module = getattr(tfa, module_name)
                Model = getattr(model_module, model_class_name)
                preprocess_module = getattr(tfa, preprocess_name)
                preprocess_fn = getattr(preprocess_module, 'preprocess_input')
                
                if not callable(Model):
                    raise TypeError(f"Expected {model_class_name} to be callable, got {type(Model)}")
                if not callable(preprocess_fn):
                    raise TypeError(f"Expected preprocess_input to be callable, got {type(preprocess_fn)}")
                    
                return Model, preprocess_fn
            except (AttributeError, TypeError) as e:
                print(f"⚠ {model_name} not available in this TensorFlow version. Using ResNet50 instead.")
                from tensorflow.keras.applications.resnet50 import ResNet50 as Model
                from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_fn
                return Model, preprocess_fn
        
        # Standard handling for other models
        else:
            model_module = getattr(tfa, module_name)
            Model = getattr(model_module, model_class_name)
            preprocess_module = getattr(tfa, preprocess_name)
            preprocess_fn = getattr(preprocess_module, 'preprocess_input')
            
            if not callable(Model):
                raise TypeError(f"Expected {model_class_name} to be callable, got {type(Model)}")
            if not callable(preprocess_fn):
                raise TypeError(f"Expected preprocess_input to be callable, got {type(preprocess_fn)}")
                
            return Model, preprocess_fn
            
    except (AttributeError, TypeError) as e:
        print(f"⚠ Error loading {model_name}: {e}. Using ResNet50 instead.")
        from tensorflow.keras.applications.resnet50 import ResNet50 as Model
        from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_fn
        return Model, preprocess_fn

def build_model_cnn():
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
      
        # Block 1
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SpatialDropout2D(0.1),

        # Block 2
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SpatialDropout2D(0.1),

        # Block 3
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SpatialDropout2D(0.15),

        # Block 4
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SpatialDropout2D(0.15),

        # Block 5
        layers.Conv2D(512, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SpatialDropout2D(0.2),

        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
               

        layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",tf.keras.metrics.AUC(name="auc")
                ]
    )
    return model

def build_model(backbone_fn, learning_rate_dl, weight_decay_dl):
    """
    Build a transfer learning model using a backbone (pre-trained model).
    
    Parameters:
    - backbone_fn: The model function/class (e.g., ResNet50, DenseNet169, etc.) - must be callable
    - learning_rate_dl: Learning rate for optimizer
    - weight_decay_dl: Weight decay for regularization
    """
    
    # Validation - ensure we got a callable (class or function)
    if backbone_fn is None:
        raise TypeError("backbone_fn cannot be None")
    
    if not callable(backbone_fn):
        raise TypeError(f"backbone_fn must be callable. Got {type(backbone_fn)}: {backbone_fn}")
    
    # Check if it's a preprocessing function (by name)
    if hasattr(backbone_fn, '__name__'):
        fn_name = backbone_fn.__name__
        if 'preprocess' in fn_name.lower():
            raise TypeError(f"Error: Got preprocessing function '{fn_name}' instead of model. "
                          "Check that load_model_and_preprocess() returns (Model_callable, preprocess_fn) in correct order")

    model_base = backbone_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    model_base.trainable = False  # Freeze base layers

    # Build transfer learning model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        model_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid")
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_dl,
            weight_decay=weight_decay_dl
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy", tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model
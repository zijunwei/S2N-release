def set_readable_param_names(model, prefix=None):
    for name, _ in model.named_parameters():
        if prefix is None:
            getattr(model, name).readable_name = name
        else:
            getattr(model, name).readable_name = '{:s}-{:s}'.format(prefix, name)

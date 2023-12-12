This folder contains ViNT and NoMaD model definitions adapted from https://github.com/robodhruv/visualnav-transformer.

These model definitions must be in specific `vint_train.models` path otherwise `torch.load` will fail with
`ModuleNotFoundError` as these models were fully serialized and not just state dictionaries.

Training of these models is not adapted to this code base yet.
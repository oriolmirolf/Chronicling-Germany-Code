"""Model to train kraken OCR model."""
import glob
import os
from pprint import pprint

import torch
from kraken.lib import default_specs                            # pylint: disable=no-name-in-module
from kraken.lib.train import RecognitionModel, KrakenTrainer    # pylint: disable=no-name-in-module
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')


def main(name: str) -> None:
    """
    Trains a OCR model.

    Args:
        name: of the model
    """
    # check for cuda device
    if torch.cuda.is_available():
        print("CUDA device is available!")
    else:
        print("CUDA device is not available.")

    # create folder for model saves
    os.makedirs(f'models/{name}', exist_ok=False)

    # create training- and evaluation set
    training_files = list(glob.glob('data/train/*.xml'))
    evaluation_files = list(glob.glob('data/valid/*.xml'))

    # set some hyperparameters
    hparams = default_specs.RECOGNITION_HYPER_PARAMS.copy()
    hparams['epochs'] = 100
    hparams['lrate'] = 0.001
    hparams['warmup'] = 1
    hparams['augment'] = True
    hparams['batch_size'] = 128
    hparams['freeze_backbone'] = 2
    # hparams['schedule'] = 'exponential'
    # hparams['step_size'] = 100,
    # hparams['gamma'] = 0.02,

    # init model
    model = RecognitionModel(hyper_params=hparams,
                             output=f'models/{name}/model',
                             model='load_model/german_newspapers_kraken.mlmodel',
                             num_workers=23,
                             training_data=training_files,
                             evaluation_data=evaluation_files,
                             resize='new',
                             format_type='page')

    # print hyperparameter of model
    pprint(f"{model.hparams}")

    # init logger and training
    logger = TensorBoardLogger("logs", name=name)
    trainer = KrakenTrainer(pl_logger=logger)               # pylint: disable=unexpected-keyword-arg

    # start training
    trainer.fit(model)


if __name__ == '__main__':
    main(name='run12')

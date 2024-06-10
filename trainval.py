import os
import argparse
import baseline
from SingularTrajectory import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/singulartrajectory-transformerdiffusion-zara1.json", type=str, help="config file path")
    parser.add_argument('--tag', default="SingularTrajectory-TEMP", type=str, help="personal tag for the model")
    parser.add_argument('--gpu_id', default="0", type=str, help="gpu id for the model")
    parser.add_argument('--test', default=False, action='store_true', help="evaluation mode")
    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args))

    print("===== Configs =====")
    hyper_params = get_exp_config(args.cfg)
    print_arguments(hyper_params)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
    hook_func = DotDict({"model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
                         "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
                         "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook})
    ModelTrainer = getattr(trainer, *[s for s in trainer.__dict__.keys() if hyper_params.baseline in s.lower()])
    trainer = ModelTrainer(base_model=PredictorModel, model=SingularTrajectory, hook_func=hook_func,
                           args=args, hyper_params=hyper_params)

    if not args.test:
        trainer.init_descriptor()
        trainer.fit()
    else:
        trainer.load_model()
        print("Testing...", end=' ')
        results = trainer.test()
        print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])

"""
TRAIN TUNE SKIPGANOMALY/WGANOMALY

. Example: Run the following command from the terminal.
    run train_tune.py                                    \
        --model <skipganomaly, w_skipganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
        --tune TRUE
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
from ray import tune
from functools import partial
import os
from ray.tune.suggest import Repeater
from ray.tune.suggest.basic_variant import BasicVariantGenerator
import copy
##

def generate_variants(config, num_seeds=1, num_samples=1, seed_generator = lambda seed: seed):
    """
    Requires all elements to have a .sample method, like tune.choice etc. 
    cannot be grid_search.
    Given a config of hyperparameters, generates num_samples of them by 
    calling .sample on each hyperparameter seach space. Then makes 
    num_seeds copies of these, each with a different random seed.
    """
    variants = []
    for sample in range(num_samples):
        variant = {param:space.sample() for param, space in config.items()}
        for seed in range(num_seeds):
            seeded_variant = copy.deepcopy(variant)
            seeded_variant['seed'] = seed_generator(seed)
            variants.append(seeded_variant)
    return tune.grid_search(variants)





options = Options().parse()
dades = load_data(options)

def main(config, data=None):
    """ Training
    """
    opt = options
    print(opt.w_con)
    #opt.lr = config["lr"]
    opt.w_adv = config["w_latadv"]
    opt.w_con = config["w_con"]
    opt.w_lat = config["w_latadv"]
    #opt.nz = config["nz"]
    
    print(opt.w_con)
    #data = dades
    model = load_model(opt, data)
    model.train()

    
    
if __name__ == '__main__':
    #main()
    #options = Options().parse()
    #search = BasicVariantGenerator()
    #re_search_alg = Repeater(searcher=search, repeat=3)
    #config= generate_variants(config, num_seeds=3, num_samples=2)
   
    #TUNING EXAMPLE, LEARNING RATE, WEIGHTS OF LOSSES, LATENT CRITIC SPACE...
    analysis = tune.run(
        tune.with_parameters(main,data=dades),
        num_samples=2,
        resources_per_trial={'gpu': 1},
        #search_alg=re_search_alg,
        config={
            #"lr":tune.grid_search([0.0002,0.0004]),
            "w_latadv": tune.grid_search([0.5,1,2]),
            "w_con": tune.grid_search([25,50,75]),
            #"w_lat": tune.grid_search([0.5,1,2]),
            #"nz": tune.grid_search([100,200,300])})
        })
    
    
    print("Best config: ", analysis.get_best_config( 
        metric="score", mode="max"))
    #print(analysis.results_df)
 
    
   

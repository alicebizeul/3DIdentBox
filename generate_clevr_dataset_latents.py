"""Create latents for 3DIdent dataset."""

import sys
sys.path.append('../../')
import os
import numpy as np
import spaces
import latent_spaces
import argparse
import spaces_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", default=1000000, type=int)
    parser.add_argument("--n-objects", default=1, type=int)
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--rotation-and-color-only", action="store_true")
    parser.add_argument("--rotation-only", action="store_true")
    parser.add_argument("--color-only", action="store_true")
    parser.add_argument("--fixed-spotlight", action="store_true")
    parser.add_argument("--non-periodic-rotation-and-color", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--mi", action="store_true")
    parser.add_argument("--basic", action="store_true")
    parser.add_argument("--all-hues", action="store_true")
    parser.add_argument("--all-positions", action="store_true")
    parser.add_argument("--all-rotations", action="store_true")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--debug2",action="store_true")
    parser.add_argument("--debug3",action="store_true")
    parser.add_argument("--first_content", action="store_true")

    args = parser.parse_args()

    print(args)

    assert not (
        args.position_only and args.rotation_and_color_only
    ), "Only either position-only or rotation-and-color-only can be set"

    os.makedirs(args.output_folder, exist_ok=True)

    """
    render internally assumes the variables form these value ranges:
    
    per object:
        0. x position in [-3, -3]
        1. y position in [-3, -3]
        2. z position in [-3, -3]
        3. alpha rotation in [0, 2pi]
        4. beta rotation in [0, 2pi]
        5. gamma rotation in [0, 2pi]
        6. theta spot light in [0, 2pi]
        7. hue object in [0, 2pi]
        8. hue spot light in [0, 2pi]
    
    per scene:
        9. hue background in [0, 2pi]
    """

    n_angular_variables = args.n_objects * 6 + 1
    n_non_angular_variables = args.n_objects * 3

    if args.non_periodic_rotation_and_color:
        print("Got in the first loop")
        if args.mi:
            s = latent_spaces.ProductLatentSpace(
                [
                    latent_spaces.LatentSpace(
                        spaces.NBoxSpace(n_non_angular_variables + n_angular_variables-3),
                        lambda space, size, device: space.uniform(size, device=device),
                        lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                    ),
                    latent_spaces.LatentSpace(
                        spaces.NBoxSpace(3,min_=-1.0,max_=1.0),   # original -0.25, 0.25
                        lambda space, size, device: space.uniform(size, device=device),
                        lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                    ),
                ]
            )
        elif args.basic:
            s = latent_spaces.ProductLatentSpace(
                [   latent_spaces.LatentSpace(
                        spaces.NBoxSpace(3,min_=-1.0,max_=1.0),   # original -0.25, 0.25
                        lambda space, size, device: space.uniform(size, device=device),
                        lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                    ),
                    latent_spaces.LatentSpace(
                        spaces.NBoxSpace(n_non_angular_variables + n_angular_variables-3),
                        lambda space, size, device: space.uniform(size, device=device),
                        lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                    ),
                ]
            )
        elif args.multimodal and args.all_hues:
            if args.first_content:
                s = latent_spaces.ProductLatentSpace(
                    [
                        # Positions
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # Rotations
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),                        
                        ##### rotation angle fixed here
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                        # spotlight position
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                        # Hues
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                    ]
                )
            else:
                s = latent_spaces.ProductLatentSpace(
                    [
                        # Positions
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                        # Rotations
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),                        
                        ##### rotation angle fixed here
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),

                        # spotlight position
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # Hues
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                    ]
                )
        elif args.multimodal and args.all_positions:
            if args.first_content:
                s = latent_spaces.ProductLatentSpace(
                    [
                        # Positions 
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device),
                        ),                    
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        # Rotations
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),                        
                        ##### rotation angle fixed here
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # spotlight position
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # Hues
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                    ]
                )
            else:
                s = latent_spaces.ProductLatentSpace(
                    [
                        # Positions 
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device),
                        ),                    
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        # Rotations
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),                        
                        ##### rotation angle fixed here
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                        # spotlight position
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                        # Hues
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                    ]
                )
        elif args.multimodal and args.all_rotations:
            if args.first_content:
                s = latent_spaces.ProductLatentSpace(
                    [
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device=device),
                        ),
                        # removed second rotation
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                    ]
                )
            else: 
                s = latent_spaces.ProductLatentSpace(
                    [
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device=device),
                        ),
                        # removed second rotation
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                    ]
                )

        elif args.debug:  # spotlight limited

            s = latent_spaces.ProductLatentSpace(
                    [
                        # Positions 
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device),
                        ),                    
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        # Rotations
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),                        
                        ##### rotation angle fixed here
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # spotlight position
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1,-1,0),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # Hues
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                    ]
                )

            # s = latent_spaces.ProductLatentSpace(
            #     [
            #         latent_spaces.LatentSpace(
            #             spaces.NBoxSpace(3),
            #             lambda space, size, device: space.uniform(size, device=device),
            #             lambda space, mean, std, size, device: space.normal(mean, std, size, device),
            #         ),
            #         latent_spaces.LatentSpace(
            #             spaces.NBoxSpace(n_non_angular_variables + n_angular_variables-3-3),
            #             lambda space, size, device: space.uniform(size, device=device),
            #             lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
            #         ),
            #         latent_spaces.LatentSpace(
            #             spaces.NBoxSpace(1),
            #             lambda space, size, device: space.uniform(size, device=device),
            #             lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
            #         ),
            #         latent_spaces.LatentSpace(
            #             spaces.NBoxSpace(1),
            #             lambda space, size, device: space.uniform(size, device=device),
            #             lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
            #         ),
            #         latent_spaces.LatentSpace(
            #             spaces.NBoxSpace(1),
            #             lambda space, size, device: space.uniform(size, device=device),
            #             lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
            #         ),
            #     ]
            # )

        elif args.debug2:

            s = latent_spaces.ProductLatentSpace(
                    [
                        # Positions 
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device),
                        ),                    
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        # Rotations
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),                        
                        ##### rotation angle fixed here
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # spotlight position
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # Hues
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                    ]
                )

        elif args.debug3:

            s = latent_spaces.ProductLatentSpace(
                    [
                        # Positions 
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.uniform(size, device),
                        ),                    
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        # Rotations
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1,0,1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),                     
                        ##### rotation angle fixed here
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1,-1,0),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ), 
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.delta(size, device=device),
                            lambda space, mean, std, size, device: space.delta(size, device=device),
                        ),
                        # spotlight position
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(1),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.normal(mean, std, size, device),
                        ),
                        # Hues
                        latent_spaces.LatentSpace(
                            spaces.NBoxSpace(3),
                            lambda space, size, device: space.uniform(size, device=device),
                            lambda space, mean, std, size, device: space.trunc_normal(mean, std, size, device),
                        ),
                    ]
                )
            


        else:
            s = latent_spaces.LatentSpace(
                spaces.NBoxSpace(n_non_angular_variables + n_angular_variables),
                lambda space, size, device: space.uniform(size, device=device),
                None,
            )
            

    else:
        s = latent_spaces.ProductLatentSpace(
            [
                latent_spaces.LatentSpace(
                    spaces.NBoxSpace(n_non_angular_variables),
                    lambda space, size, device: space.uniform(size, device=device),
                    None,
                ),
                latent_spaces.LatentSpace(
                    spaces.NSphereSpace(n_angular_variables + 1),
                    lambda space, size, device: space.uniform(size, device=device),
                    None,
                ),
            ]
        )

    if args.deterministic: 
        if args.mi:
            raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
            raw_latents_view2 = s.sample_conditional(raw_latents_view1,[0.0,1.0], size=int(args.n_points/2), device="cpu").numpy()   # [1.0,0.5] for mi originally 
            raw_latents_view1 = raw_latents_view1.numpy()
            #if args.all_hues: 
            #    raw_latents_view2[:7] = raw_latents_view1[:7]
            #else: raise("Other data augmentations are not yet supported")

            raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)
        elif args.basic:
            raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
            raw_latents_view2 = s.sample_conditional(raw_latents_view1,[1.0,0.0], size=int(args.n_points/2), device="cpu").numpy()   # [1.0,0.5] for mi originally 
            raw_latents_view1 = raw_latents_view1.numpy()
            #if args.all_hues: 
            #    raw_latents_view2[:7] = raw_latents_view1[:7]
            #else: raise("Other data augmentations are not yet supported")

            raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)

        elif args.multimodal and args.all_hues:
            if args.first_content:
                raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
                raw_latents_view2 = s.sample_conditional(raw_latents_view1,[0.0,1.0,None,1.0,1.0,None,None,None], size=int(args.n_points/2), device="cpu").numpy()
                raw_latents_view1 = raw_latents_view1.numpy()
                raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)
            else:
                raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
                raw_latents_view2 = s.sample_conditional(raw_latents_view1,[1.0,0.0,None,0.0,0.0,None,None,None], size=int(args.n_points/2), device="cpu").numpy()
                raw_latents_view1 = raw_latents_view1.numpy()
                raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)


        elif args.multimodal and args.all_positions:
            if args.first_content:
                raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
                raw_latents_view2 = s.sample_conditional(raw_latents_view1,[None,None,None,0.0,None,0.0,0.0,1.0], size=int(args.n_points/2), device="cpu").numpy()
                raw_latents_view1 = raw_latents_view1.numpy()
                raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)
            else:
                raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
                raw_latents_view2 = s.sample_conditional(raw_latents_view1,[None,None,None,1.0,None,1.0,1.0,0.0], size=int(args.n_points/2), device="cpu").numpy()
                raw_latents_view1 = raw_latents_view1.numpy()
                raw_latents = np.append(raw_latents_view1,raw_latents_view2,0) 


        elif args.multimodal and args.all_rotations:
            if args.first_content:
                raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
                raw_latents_view2 = s.sample_conditional(raw_latents_view1,[0.0,None,None,None,None,1.0], size=int(args.n_points/2), device="cpu").numpy()
                raw_latents_view1 = raw_latents_view1.numpy()
                raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)
            else:
                raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
                raw_latents_view2 = s.sample_conditional(raw_latents_view1,[1.0,None,None,None,None,0.0], size=int(args.n_points/2), device="cpu").numpy()
                raw_latents_view1 = raw_latents_view1.numpy()
                raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)

        elif args.debug:
            raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
            raw_latents_view2 = s.sample_conditional(raw_latents_view1,[None,None,None,0.0,None,0.0,0.0,1.0], size=int(args.n_points/2), device="cpu").numpy()
            raw_latents_view1 = raw_latents_view1.numpy()
            raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)
        
        elif args.debug2:
            raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
            raw_latents_view2 = s.sample_conditional(raw_latents_view1,[None,None,None,None,None,0.0,0.0,1.0], size=int(args.n_points/2), device="cpu").numpy()
            raw_latents_view1 = raw_latents_view1.numpy()
            raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)

        elif args.debug3: 
            raw_latents_view1 = s.sample_marginal(int(args.n_points/2), device="cpu")
            raw_latents_view2 = s.sample_conditional(raw_latents_view1,[None,None,None,0.0,0.0,None,0.0,1.0], size=int(args.n_points/2), device="cpu").numpy()
            raw_latents_view1 = raw_latents_view1.numpy()
            raw_latents = np.append(raw_latents_view1,raw_latents_view2,0)
         
        else: raw_latents = s.sample_marginal(args.n_points, device="cpu").numpy()

    if args.position_only or args.rotation_and_color_only:
        assert args.n_objects == 1, "Only one object is supported for fixed variables"

    if args.non_periodic_rotation_and_color:
        if args.position_only:
            raw_latents[:, n_non_angular_variables:] = np.array(
                [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
            )
        if args.rotation_and_color_only or args.rotation_only or args.color_only:
            raw_latents[:, :n_non_angular_variables] = np.array([0, 0, 0])
        if args.rotation_only:
            # additionally fix color
            raw_latents[:, -3:] = np.array([-1, 0, 1.0])
        if args.color_only:
            # additionally fix rotation
            raw_latents[
                :, n_non_angular_variables : n_non_angular_variables + 4
            ] = np.array([-1, -0.5, 0.5, 1.0])

        if args.fixed_spotlight:
            # assert not args.rotation_only
            raw_latents[:, [-2, -4]] = np.array([0.0, 0.0])


        # the raw latents will later be used for the sampling process
        if args.deterministic:
            if args.all_hues:
                np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.all_positions:
                np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.all_rotations:
                np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.debug:
                np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.debug2:
                np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.debug3:
                np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            else:raise("Other data augmentations are not yet supported")
        else:np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)

        # get rotation and color latents from large vector
        rotation_and_color_latents = raw_latents[:, n_non_angular_variables:]
        rotation_and_color_latents *= np.pi #/2

        # could change this
        position_latents = raw_latents[:, :n_non_angular_variables]
        position_latents *= 2   
    else:
        if args.position_only:
            spherical_fixed_angular_variables = np.array(
                [np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 2, 0, 1.5 * np.pi]
            )
            cartesian_fixed_angular_variables = spaces_utils.spherical_to_cartesian(
                1, spherical_fixed_angular_variables
            )
            raw_latents[:, n_non_angular_variables:] = cartesian_fixed_angular_variables
        if args.rotation_and_color_only:
            fixed_non_angular_variables = np.array([0, 0, 0])
            raw_latents[:, :n_non_angular_variables] = fixed_non_angular_variables

        if args.deterministic:
            if args.all_hues:np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.all_positions:np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.all_rotations:np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.debug: np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.debug2: np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            elif args.debug3: np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)
            else:raise("Other data augmentations are not yet supported")
        else:np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)

        # convert angular latents from cartesian to angular representation
        rotation_and_color_latents = spaces_utils.cartesian_to_spherical(
            raw_latents[:, n_non_angular_variables:]
        )[1]
        # map all but the last latent from [0,pi] to [0, 2pi]
        rotation_and_color_latents[:, :-1] *= 2

        position_latents = raw_latents[:, :n_non_angular_variables]
        # map z coordinate from -1,+1 to 0,+1
        position_latents[:, 2:n_non_angular_variables:3] = (
            position_latents[:, 2:n_non_angular_variables:3] + 1
        ) / 2.0
        position_latents *= 3

    latents = np.concatenate((position_latents, rotation_and_color_latents), 1)

    reordered_transposed_latents = []
    for n in range(args.n_objects):
        reordered_transposed_latents.append(latents.T[n * 3 : n * 3 + 3])
        reordered_transposed_latents.append(latents.T[n_non_angular_variables + n * 6 : n_non_angular_variables + n * 6 + 6])

    reordered_transposed_latents.append(latents.T[-1].reshape(1, -1))
    reordered_latents = np.concatenate(reordered_transposed_latents, 0).T

    # the latents will be used by the rendering process to generate the images
    if args.deterministic:
        if args.all_hues:
            np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)
        elif args.all_positions:
            np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)
        elif args.all_rotations:
            np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)
        elif args.debug: 
            np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)
        elif args.debug2: 
            np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)
        elif args.debug3: 
            np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)
        else:raise("Other data augmentations are not yet supported")
    else:np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)

    print('Size of the latents', reordered_latents.shape)


if __name__ == "__main__":
    main()

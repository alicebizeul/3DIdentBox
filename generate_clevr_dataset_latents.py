"""Create latents for 3DIdent dataset.
This code builds on the following projects:
- https://github.com/brendel-group/cl-ica
- https://github.com/ysharma1126/ssl_identifiability
"""
import sys
sys.path.append('../../')
import os
import torch
import spaces
import latent_spaces
import argparse
import numpy as np
import pandas as pd

# make sure commands are consistent
# only one object for fixed position
# make sure the conditional multinomial gives right values
# no multinomial noise is one object
# max classes object 8

def main():
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument("--n-pairs", default=1000000, type=int)
    parser.add_argument("--n-objects", default=1, type=int)
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--causal", action="store_true")

    # factors of variations
    parser.add_argument("--object", action="store_true")
    parser.add_argument("--position", action="store_true")
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--hue", action="store_true")

    # structural model
    parser.add_argument("--object-content", action="store_true")
    parser.add_argument("--object-style", action="store_true")
    parser.add_argument("--object-ms", action="store_true")
    parser.add_argument("--position-content", action="store_true")
    parser.add_argument("--position-style", action="store_true")
    parser.add_argument("--position-ms", action="store_true")
    parser.add_argument("--rotation-content", action="store_true")
    parser.add_argument("--rotation-style", action="store_true")
    parser.add_argument("--rotation-ms", action="store_true")
    parser.add_argument("--hue-content", action="store_true")
    parser.add_argument("--hue-style", action="store_true")
    parser.add_argument("--hue-ms", action="store_true")

    # causal relationships
    parser.add_argument("--intra-content", action="store_true")
    parser.add_argument("--intra-style", action="store_true")
    parser.add_argument("--inter-content-style", action="store_true")

    # generative parameters
    parser.add_argument("--min", type=float, default=-1.0)
    parser.add_argument("--max", type=float, default=1.0)
    parser.add_argument("--continuous-marginal", type=str, default="uniform")
    parser.add_argument("--continuous-conditional", type=str, default="normal")
    parser.add_argument("--normal-marginal-std", type=float, default=1.0)
    parser.add_argument("--normal-conditional-std", type=float, default=1.0)
    parser.add_argument("--normal-conditional-noise", type=float, default=1.0)
    parser.add_argument("--uniform-marginal-a", type=float, default=-1.0)
    parser.add_argument("--uniform-marginal-b", type=float, default=1.0)
    parser.add_argument("--uniform-conditional-a", type=float, default=-0.1)
    parser.add_argument("--uniform-conditional-b", type=float, default=0.1)
    parser.add_argument("--uniform-conditional-noise-a", type=float, default=-0.1)
    parser.add_argument("--uniform-conditional-noise-b", type=float, default=0.1)
    parser.add_argument("--multinomial-noise",type=int,default=3)

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"m1"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"m2"), exist_ok=True)

    # create model partition
    latent_list={"content":{},"style":{},"ms":{}}
    if args.object: 
        gen_type = ["content","style","ms"][np.argmax(1*[args.object_content,args.object_style,args.object_ms])]
        latent_list[gen_type]["object"] = None  
    if args.position: 
        gen_type = ["content","style","ms"][np.argmax(1*[args.position_content,args.position_style,args.position_ms])]
        latent_list[gen_type]["position_x"] = None
        latent_list[gen_type]["position_y"] = None
        latent_list[gen_type]["position_z"] = None
    if args.rotation: 
        gen_type = ["content","style","ms"][np.argmax(1*[args.rotation_content,args.rotation_style,args.rotation_ms])]
        latent_list[gen_type]["rotation_object_alpha"] = None
        latent_list[gen_type]["rotation_object_beta"] = None
        latent_list[gen_type]["rotation_spot"] = None
    if args.hue: 
        gen_type = ["content","style","ms"][np.argmax(1*[args.hue_content,args.hue_style,args.hue_ms])]
        latent_list[gen_type]["object_hue"] = None
        latent_list[gen_type]["back_hue"] = None
        latent_list[gen_type]["spot_hue"] = None

    # create generative model
    latent_spaces_list={}
    params_marginal = {}
    params_conditional = {}
    for part in ["content","style","ms"]:
        if part != "ms": idx_marg, idx_cond = ("normal" if args.continuous_marginal == "normal" else "uniform"), ("normal" if args.continuous_conditional == "normal" else "uniform")
        else: idx_marg, idx_cond = "delta", "delta"
        dist = {"normal": lambda space, mean, params, size, device: space.normal(mean, params["std"], size, device),
                "uniform" : lambda space, mean, params, size, device: space.uniform(mean, params["a"], params["b"], size, device),
                "multinomial" : lambda space, mean, params, size, device: space.multinomial(mean, params["classes"], size, weights=params["weights"], uniform=params["uniform"],device=device),
                "delta":lambda space, mean, params, size, device: space.delta(size, device=device),}
        
        marginal, conditional = dist[idx_marg], dist[idx_cond]

        for k in list(latent_list[part].keys()):
            if k != "object":latent_list[part][k]=latent_spaces.LatentSpace(spaces.NBoxSpace(1,min_=args.min,max_=args.max),marginal,conditional)
            else: latent_list[part][k]=latent_spaces.LatentSpace(
                                    spaces.NBoxSpace(1,min_=args.min,max_=args.max),
                                    lambda space, mean, params, size, device: space.multinomial(mean, params["classes"], size, weights=params["weights"], uniform=params["uniform"],device=device),
                                    lambda space, mean, params, size, device: space.multinomial(mean, params["classes"], size, weights=params["weights"], uniform=params["uniform"],device=device))

            nb_instances = 1 if ((k not in ["object","position_x","position_y","position_z","rotation_object_alpha","rotation_object_beta","object_hue"]) or (args.n_objects ==1)) else args.n_objects
            for tmp_idx in range(nb_instances):
                latent_spaces_list[k+"_object_"+str(tmp_idx)]=latent_list[part][k]
                params_marginal[k+"_object_"+str(tmp_idx)]={"uniform":{"a":args.uniform_marginal_a,"b":args.uniform_marginal_b},
                                        "normal":{"std":args.normal_marginal_std},
                                        "multinomial":{"classes":args.n_objects,"uniform":True,"weights":None},
                                        "delta":{}}[idx_marg if k != "object" else "multinomial"]
                params_conditional[k+"_object_"+str(tmp_idx)]={"uniform":{"a":args.uniform_conditional_a if part=="content" else args.uniform_conditional_noise_a,"b":args.uniform_conditional_b if part=="content" else args.uniform_conditional_noise_b},
                                        "normal":{"std":args.normal_conditional_std if part=="content" else args.normal_conditional_noise},
                                        "multinomial":{"classes":args.multinomial_noise,"uniform":True,"weights":None},
                                        "delta":{}}[idx_cond if k != "object" else "multinomial"]

    # reorder: depending on what's fixed: scene hue, scene rotation, object hue, object rotation, object position, object type
    s = latent_spaces.ProductLatentSpace(list(latent_spaces_list.values())) # add ordered list
    params_marginal={k: v for k,v in zip(np.arange(len(list(latent_spaces_list.values()))),list(params_marginal.values()))}
    params_conditional={k:v for k,v in zip(np.arange(len(list(latent_spaces_list.values()))),list(params_conditional.values()))}

    # generating latents - causal dep or not
    if args.causal:
        raise NotImplementedError
    else:
        raw_latents_view1 = s.sample_marginal(means=torch.zeros([args.n_pairs,len(latent_spaces_list)]),params=params_marginal,size=args.n_pairs, device="cpu")
        raw_latents_view2 = pd.DataFrame(s.sample_conditional(means=raw_latents_view1,params=params_conditional,size=args.n_pairs, device="cpu").numpy(),columns=list(latent_spaces_list.keys()))
        raw_latents_view1 = pd.DataFrame(raw_latents_view1.numpy(),columns=list(latent_spaces_list.keys()))

    # add fixed variables 
    columns=[]
    fixed_values=[]
    if not args.hue:
        columns.extend(["spot_hue_object_0","back_hue_object_0"]+[f"object_hue_object_{k}" for k in range(args.n_objects)])
        fixed_values.append(0.0*np.ones([args.n_pairs,1]))
        fixed_values.append(1.0*np.ones([args.n_pairs,1]))
        for _ in range(args.n_objects):fixed_values.append(-1.0*np.ones([args.n_pairs,1]))
    if not args.rotation:
        columns.extend([f"rotation_object_alpha_object_{k}" for k in range(args.n_objects)]+[f"rotation_object_beta_object_{k}" for k in range(args.n_objects)]+["rotation_spot_object_0"])
        for _ in range(args.n_objects):fixed_values.append(1.0*np.ones([args.n_pairs,1]))
        for _ in range(args.n_objects):fixed_values.append(-1.0*np.ones([args.n_pairs,1]))
        for _ in range(args.n_objects):fixed_values.append(-0.5*np.ones([args.n_pairs,1]))
    if not args.position:
        columns.extend([f"position_x_object_{k}" for k in range(args.n_objects)]+[f"position_y_object_{k}" for k in range(args.n_objects)]+[f"position_z_object_{k}" for k in range(args.n_objects)])
        for _ in range(3*args.n_objects):fixed_values.append(np.zeros([args.n_pairs,1]))
    if not args.object: 
        columns.extend([f"object_object_{k}" for k in range(args.n_objects)])
        for _ in range(args.n_objects):fixed_values.append(np.zeros([args.n_pairs,1]))
    columns, fixed_values = columns, np.asarray(fixed_values)
    raw_latents_view1=pd.concat([raw_latents_view1,pd.DataFrame(fixed_values,columns)],axis=1)
    raw_latents_view2=pd.concat([raw_latents_view2,pd.DataFrame(fixed_values,columns)],axis=1)

    # reorder latents
    static_list=[]
    static_list.extend(["spot_hue_object_0","back_hue_object_0","rotation_spot_object_0"])
    static_list.extend(list(filter(lambda x: x.startswith('object_hue'), list(latent_spaces_list.keys()))))
    static_list.extend(list(filter(lambda x: x.startswith('rotation_object'), list(latent_spaces_list.keys()))))
    static_list.extend(list(filter(lambda x: x.startswith('position'), list(latent_spaces_list.keys()))))
    static_list.extend(list(filter(lambda x: x.startswith('object_object'), list(latent_spaces_list.keys()))))
    raw_latents_view1=raw_latents_view1.reindex(columns=static_list)
    raw_latents_view2=raw_latents_view2.reindex(columns=static_list)

    # save raw latents 
    np.save(os.path.join(args.output_folder, "m1","raw_latents.npy"), raw_latents_view1.to_numpy())
    np.save(os.path.join(args.output_folder, "m2","raw_latents.npy"), raw_latents_view2.to_numpy())

    # raw latents to latents: rotation
    rot_cols = raw_latents_view1.filter(like='rotation')
    raw_latents_view1[rot_cols.columns]=np.pi*rot_cols
    rot_cols = raw_latents_view2.filter(like='rotation')
    raw_latents_view2[rot_cols.columns]=np.pi*rot_cols

    # raw latents to latents: hues      
    hue_cols = raw_latents_view1.filter(like='hue')
    raw_latents_view1[hue_cols.columns]=np.pi*hue_cols
    hue_cols = raw_latents_view2.filter(like='hue')
    raw_latents_view2[hue_cols.columns]=np.pi*hue_cols

    # raw latents to latents: position     
    pos_cols = raw_latents_view1.filter(like='position')
    raw_latents_view1[pos_cols.columns]=2*pos_cols
    pos_cols = raw_latents_view2.filter(like='position')
    raw_latents_view2[pos_cols.columns]=2*pos_cols

    # save Blender latents
    np.save(os.path.join(args.output_folder, "m1","latents.npy"), raw_latents_view1.to_numpy())
    np.save(os.path.join(args.output_folder, "m2","latents.npy"), raw_latents_view2.to_numpy())

if __name__ == "__main__":
    main()

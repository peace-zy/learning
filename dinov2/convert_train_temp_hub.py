import torch
import json
from omegaconf import OmegaConf
from dinov2.models import build_model_from_cfg
from dinov2.configs import dinov2_default_config


def main():
    config_file = "checkpoints/dinov2-vitb14-bs1024-ep100-a800/config_m.yaml"
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(config_file)
    #cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))

    student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)


    hub_model = torch.load("/data/nfs-ten9/nfs/zhangyan461/models/dinov2_vitb14_pretrain.pth")
    hub_model["pos_embed"] = hub_model["pos_embed"][:, 0:student_backbone.state_dict()["pos_embed"].shape[1], :]
    #student_backbone.load_state_dict(hub_model, strict=False)
    hub_keys = list(hub_model.keys())

    #dinov2_model = torch.load("checkpoints/dinov2-vitb14-bs1024-ep100-a800/eval/training_12499/teacher_checkpoint.pth")
    dinov2_model = torch.load("checkpoints/dinov2-vitb14-bs1024-ep100-a800-mlp/eval/training_12499/teacher_checkpoint.pth")
    dinov2_state_dict = dinov2_model["teacher"]
    # prefix backbone
    dinov2_keys = list(dinov2_state_dict.keys())
    # hub 0 -> train 0.0
    train_keys = list(student_backbone.state_dict().keys())
    # dinov2->train->hub
    map_dict = {
            "dinov2_to_train": dict(zip(dinov2_keys, train_keys)),
            "train_to_hub": dict(zip(train_keys, hub_keys)),
            "hub_to_train": dict(zip(hub_keys, train_keys))
            }

    with open("dinov2_vitb14_map.json", "w") as f:
        json.dump(map_dict, f, indent=4)
    for e in list(zip(hub_keys, dinov2_keys, train_keys)):
        print("{} {} \t{} {}\t{} {}".format(e[0], hub_model[e[0]].shape, e[1], dinov2_state_dict[e[1]].shape, e[2], student_backbone.state_dict()[e[2]].shape))
    #student_backbone.load_state_dict(dinov2_model["teacher"], strict=False)
    for old, new in map_dict["dinov2_to_train"].items():
        val = dinov2_state_dict.pop(old)
        dinov2_state_dict[new] = val
    student_backbone.load_state_dict(dinov2_state_dict, strict=False)
    student_state_dict = student_backbone.state_dict()
    for old, new in map_dict["train_to_hub"].items():
        val = student_state_dict.pop(old)
        student_state_dict[new] = val


    out_name = "dinov2_vitb14_offline_train.pt"
    torch.save(student_state_dict, out_name)

    model = torch.load(out_name)

if __name__ == "__main__":
    main()

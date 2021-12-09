from models.model_clam import CLAM_SB
from models.model_clam import CLAM_MB
import torch
ckpt_path = "trained_params/tcga_cptac_lung_cv/lung_public_cv_CLAM_50_s1/s_6_checkpoint.pt"
model_dict = {"dropout": False, 'n_classes': 2}


model_size = 'small'
model_type = 'clam_sb'

if model_size is not None and model_type in ['clam_sb', 'clam_mb']:
    model_dict.update({"size_arg": model_size})

model = CLAM_MB(**model_dict)


ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
ckpt_clean = {}
for key in ckpt.keys():
    if 'instance_loss_fn' in key:
        continue
    new_key = key.replace('.module', '')
    new_key = new_key.replace('attention_net.3', 'attention_net.2')
    ckpt_clean.update({new_key: ckpt[key]})

#for key, value in ckpt_clean.items() :
    #print (key)
model.load_state_dict(ckpt_clean, strict=True)

sample_input = torch.rand(size = (100, 1024))

logits, Y_prob, Y_hat, A_raw, results_dict = model(sample_input)
print(logits)
print(A_raw.shape)
#print(model)


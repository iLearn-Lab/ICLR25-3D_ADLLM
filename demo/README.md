# 3d_aff

## Demmo
```bash
# pointnet++
python /workspace/project/Aff_LLM_debug/app_openad.py --config /workspace/project/Aff_LLM_debug/models/openad/config/full_shape_cfg_modify_demo.py --checkpoint /workspace/project/Aff_LLM_debug/model_ckpt/pointnet_best_model.t7
# pointnet++,prompt,only affordancelabel
python /workspace/project/Aff_LLM_debug/app_openad.py --config /workspace/project/Aff_LLM_debug/models/openad/config/full_shape_cfg_modify_demo.py --checkpoint /workspace/project/Aff_LLM_debug/model_ckpt/label_pointnet_best_model.t7
# pointnet++,affdecoder,
python /workspace/project/Aff_LLM_debug/app_openad.py --config /workspace/project/Aff_LLM_debug/models/openad/config/full_shape_cfg_modify_samdecoder_demo.py  --checkpoint /workspace/project/Aff_LLM_debug/model_ckpt/affdecoder_best_model.t7


# dgcnn
# celoss+diceloss
python /workspace/project/Aff_LLM_debug/app_openad.py --config /workspace/project/Aff_LLM_debug/models/openad/config/full_shape_cfg_DGCNN_modify_demo.py --checkpoint /workspace/project/Aff_LLM_debug/model_ckpt/DGCNN_best_model.t7
# celoss
python /workspace/project/Aff_LLM_debug/app_openad.py --config /workspace/project/Aff_LLM_debug/models/openad/config/full_shape_cfg_DGCNN_modify_demo.py --checkpoint /workspace/project/Aff_LLM_debug/model_ckpt/DGCNN_CE_best_model.t7

```
### Train a semi-supervised model with supervised contrastive loss (SupCon).  

Update deepdisc to the latest version.

Make sure that the TRAIN_FILE annotations have a `obj_id` field that contains a unique object ID.  The code uses a `SupConDictMapper` imported from `configs/custom/demo_mappers.py` to add the IDs.  

Set the CFG_FILE to `demo_ssl_swin.py` under the `configs/solo/` directory.  In principle, we just need to use 
a new ROI head.  The config imports the new ROI head from `configs/custom/roiheads_cl_demo.py`.  

This ROI head will take all foreground proposals, and label/match them to the ground truth annotations.  Proposals matched to the same ground truth are considered positive pairs.  The InfoNCE loss is calculated per-batch and controlled with a temperature parameter.  Using this ROI head will result in a semi-supervised training, where the other supervised tasks are done in parallel with the self-supervised contrastive task.  

Submit the demo run with 

```
python run_model_job_30k_resume.py \
    --cfgfile ${CFG_FILE} \
    --train-metadata ${TRAIN_FILE} \
    --eval-metadata ${EVAL_FILE} \
    --num-gpus 4 \
    --run-name ${RUN_NAME} \
    --output-dir ${OUTPUT_DIR}
```
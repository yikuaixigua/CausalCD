# CausalCD: A Causal Graph Contrastive Learning Framework for Self-Supervised SAR Image Change Detection
The full code will be open source after the publication of the paper.


  ###  1. Basic configuration Settings 
  #### config.yaml
    GPU:
        use: True
        id:   0
    Train:    
        epochs:                30
        batchsize:             32 # batchsize
    STG:
        causal_k:       32     # confounder set size


### 2. Pseudo-Label-based Finetuning
#### finetune_causal.py
     python finetune_causal.py
     
### 3. Predict
#### predict.py
     python finetune_causal.py


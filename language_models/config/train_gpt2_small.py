# these make the total batch size be ~0.5M
# 8 batch size * 1024 block size * 6 gradaccum * 10 GPUs = 491,520



batch_size = 12  #12
block_size = 1024
gradient_accumulation_steps = 40#40

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# this makes total number of tokens be 300B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 20
eval_iters = 200 # how many samples used for calulating validation loss
log_interval = 10
ckpt_interval = 1000

# optimizer
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95 #0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
min_lr = 3e-5 
dtype = 'float32'
# use_minibatch = True
# plot_histogram = False

init_from = 'scratch'
load_iter = 0


sample_layer = ['transformer.wte.weight', 
                'transformer.h.0.attn.wq.weight', 
                'transformer.h.0.attn.wk.weight', 
                'transformer.h.0.attn.wv.weight', 
                'transformer.h.0.attn.wo.weight',
                'transformer.h.0.mlp.c_fc.weight',
                'transformer.h.0.mlp.c_proj.weight',
                'transformer.h.3.attn.wq.weight', 
                'transformer.h.3.attn.wk.weight', 
                'transformer.h.3.attn.wv.weight', 
                'transformer.h.3.attn.wo.weight',
                'transformer.h.3.mlp.c_fc.weight',
                'transformer.h.3.mlp.c_proj.weight',
                'transformer.h.7.attn.wq.weight', 
                'transformer.h.7.attn.wk.weight', 
                'transformer.h.7.attn.wv.weight', 
                'transformer.h.7.attn.wo.weight',
                'transformer.h.7.mlp.c_fc.weight',
                'transformer.h.7.mlp.c_proj.weight',
                'transformer.h.11.attn.wq.weight', 
                'transformer.h.11.attn.wk.weight', 
                'transformer.h.11.attn.wv.weight', 
                'transformer.h.11.attn.wo.weight',
                'transformer.h.11.mlp.c_fc.weight',
                'transformer.h.11.mlp.c_proj.weight',
                ]


comment = 'hessian spectrum'# only used in tensorboard and hessian class
save_dir = 'log_gpt2/'+comment
out_dir = 'out-gpt2/' +comment # save ckpt






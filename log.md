
W0407 10:12:47.115000 2651 torch/distributed/run.py:852] 
W0407 10:12:47.115000 2651 torch/distributed/run.py:852] *****************************************
W0407 10:12:47.115000 2651 torch/distributed/run.py:852] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0407 10:12:47.115000 2651 torch/distributed/run.py:852] *****************************************
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████| 313/313 [00:00<00:00, 359.02it/s]
Loading weights: 100%|███████████████████████| 313/313 [00:00<00:00, 361.80it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Training:   0%|                                       | 0/70000 [00:00<?, ?it/s]2026-04-07 10:13:02.154664: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1775556782.173781    2657 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1775556782.180648    2657 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1775556782.197549    2657 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775556782.197574    2657 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775556782.197578    2657 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775556782.197581    2657 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
[rank1]: Traceback (most recent call last):
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 549, in <module>
[rank1]:     main()
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 520, in main
[rank1]:     trainer.train(train_loader)
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 368, in train
[rank1]:     loss, metrics = self.training_step(batch)
[rank1]:                     ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 232, in training_step
[rank1]:     student_output = self.student(
[rank1]:                      ^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/parallel/distributed.py", line 1666, in forward
[rank1]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank1]:          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/parallel/distributed.py", line 1492, in _run_ddp_forward
[rank1]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/accelerate/utils/operations.py", line 819, in forward
[rank1]:     return model_forward(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/accelerate/utils/operations.py", line 807, in __call__
[rank1]:     return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank1]:                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank1]:     return func(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/models/omnivoice_small.py", line 252, in forward
[rank1]:     llm_outputs = self.llm(
[rank1]:                   ^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py", line 917, in wrapper
[rank1]:     output = func(self, *args, **kwargs)
[rank1]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/transformers/utils/output_capturing.py", line 253, in wrapper
[rank1]:     outputs = func(self, *args, **kwargs)
[rank1]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3/modeling_qwen3.py", line 436, in forward
[rank1]:     hidden_states = decoder_layer(
[rank1]:                     ^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/transformers/modeling_layers.py", line 93, in __call__
[rank1]:     return super().__call__(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3/modeling_qwen3.py", line 323, in forward
[rank1]:     hidden_states, _ = self.self_attn(
[rank1]:                        ^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3/modeling_qwen3.py", line 280, in forward
[rank1]:     attn_output, attn_weights = attention_interface(
[rank1]:                                 ^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py", line 92, in sdpa_attention_forward
[rank1]:     attn_output = torch.nn.functional.scaled_dot_product_attention(
[rank1]:                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: TypeError: scaled_dot_product_attention(): argument 'attn_mask' must be Tensor, not BlockMask
[rank0]: Traceback (most recent call last):
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 549, in <module>
[rank0]:     main()
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 520, in main
[rank0]:     trainer.train(train_loader)
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 368, in train
[rank0]:     loss, metrics = self.training_step(batch)
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 232, in training_step
[rank0]:     student_output = self.student(
[rank0]:                      ^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/parallel/distributed.py", line 1666, in forward
[rank0]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank0]:          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/parallel/distributed.py", line 1492, in _run_ddp_forward
[rank0]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/accelerate/utils/operations.py", line 819, in forward
[rank0]:     return model_forward(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/accelerate/utils/operations.py", line 807, in __call__
[rank0]:     return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank0]:                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/models/omnivoice_small.py", line 252, in forward
[rank0]:     llm_outputs = self.llm(
[rank0]:                   ^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py", line 917, in wrapper
[rank0]:     output = func(self, *args, **kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/transformers/utils/output_capturing.py", line 253, in wrapper
[rank0]:     outputs = func(self, *args, **kwargs)
[rank0]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3/modeling_qwen3.py", line 436, in forward
[rank0]:     hidden_states = decoder_layer(
[rank0]:                     ^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/transformers/modeling_layers.py", line 93, in __call__
[rank0]:     return super().__call__(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3/modeling_qwen3.py", line 323, in forward
[rank0]:     hidden_states, _ = self.self_attn(
[rank0]:                        ^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3/modeling_qwen3.py", line 280, in forward
[rank0]:     attn_output, attn_weights = attention_interface(
[rank0]:                                 ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py", line 92, in sdpa_attention_forward
[rank0]:     attn_output = torch.nn.functional.scaled_dot_product_attention(
[rank0]:                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: TypeError: scaled_dot_product_attention(): argument 'attn_mask' must be Tensor, not BlockMask
Training:   0%|                                       | 0/70000 [00:21<?, ?it/s]
[rank0]:[W407 10:13:23.082579832 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0407 10:13:23.913000 2651 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 2657 closing signal SIGTERM
E0407 10:13:24.178000 2651 torch/distributed/elastic/multiprocessing/api.py:984] failed (exitcode: 1) local_rank: 1 (pid: 2658) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/usr/local/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 991, in main
    run(args)
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 982, in run
    elastic_launch(
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 317, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
/kaggle/working/OmniVoice/omnivoice/train_distill.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-04-07_10:13:24
  host      : fd13d1440021
  rank      : 0 (local_rank: 0)
  exitcode  : -15 (pid: 2657)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 2657
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-04-07_10:13:23
  host      : fd13d1440021
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 2658)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
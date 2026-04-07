W0407 10:16:27.150000 2903 torch/distributed/run.py:852] 
W0407 10:16:27.150000 2903 torch/distributed/run.py:852] *****************************************
W0407 10:16:27.150000 2903 torch/distributed/run.py:852] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0407 10:16:27.150000 2903 torch/distributed/run.py:852] *****************************************
Loading weights: 100%|███████████████████████| 313/313 [00:00<00:00, 328.44it/s]
Loading weights: 100%|███████████████████████| 313/313 [00:00<00:00, 352.27it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Training:   0%|                                       | 0/70000 [00:00<?, ?it/s]2026-04-07 10:16:42.306563: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1775557002.326608    2909 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1775557002.332583    2909 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1775557002.348877    2909 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775557002.348906    2909 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775557002.348916    2909 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775557002.348922    2909 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 549, in <module>
[rank0]:     main()
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 520, in main
[rank0]:     trainer.train(train_loader)
[rank0]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 370, in train
[rank0]:     accelerator.backward(loss)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py", line 2852, in backward
[rank0]:     loss.backward(**kwargs)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/_tensor.py", line 630, in backward
[rank0]:     torch.autograd.backward(
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/__init__.py", line 364, in backward
[rank0]:     _engine_run_backward(
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py", line 865, in _engine_run_backward
[rank0]:     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: Found dtype Half but expected Float
[rank1]: Traceback (most recent call last):
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 549, in <module>
[rank1]:     main()
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 520, in main
[rank1]:     trainer.train(train_loader)
[rank1]:   File "/kaggle/working/OmniVoice/omnivoice/train_distill.py", line 370, in train
[rank1]:     accelerator.backward(loss)
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py", line 2852, in backward
[rank1]:     loss.backward(**kwargs)
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/_tensor.py", line 630, in backward
[rank1]:     torch.autograd.backward(
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/__init__.py", line 364, in backward
[rank1]:     _engine_run_backward(
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py", line 865, in _engine_run_backward
[rank1]:     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: RuntimeError: Found dtype Half but expected Float
Training:   0%|                                       | 0/70000 [00:07<?, ?it/s]
[rank0]:[W407 10:16:49.887484917 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0407 10:16:50.603000 2903 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 2909 closing signal SIGTERM
E0407 10:16:50.868000 2903 torch/distributed/elastic/multiprocessing/api.py:984] failed (exitcode: 1) local_rank: 1 (pid: 2910) of binary: /usr/bin/python3
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
  time      : 2026-04-07_10:16:50
  host      : fd13d1440021
  rank      : 0 (local_rank: 0)
  exitcode  : -15 (pid: 2909)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 2909
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-04-07_10:16:50
  host      : fd13d1440021
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 2910)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
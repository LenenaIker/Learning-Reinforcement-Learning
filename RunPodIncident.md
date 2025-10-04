I tried RunPod to run BO07_AsyncVectorEnv.
I created a custom template with a TensorFlow image.
I selected an A40 graphics card with 48GB of RAM. It cost €0.40/hour.

I downloaded everything I needed and, after fixing a few errors, I managed to run it.
I saw that CPU usage was 109% and GPU usage was 0%. Python did not recognize the GPU I rented, so I stopped the execution and solved the problem.
I ran it again, but I encountered a new problem that killed the threads related to AsyncVectorEnv.
I replaced it with SyncVectorEnv. It worked, but CPU usage was 40% and GPU usage was 1%. I think the GPU was waiting for the CPU to generate environments. So I put AsyncVectorEnv back in and fixed the problems. After a while, I succeeded, but CPU usage was 30% and GPU usage was 0%.


During all the changes, the game times did not drop below 2 minutes and 20 seconds, 

Keep in mind that my laptop runs the same thing in 1:58.
My laptop has a decent processor: Intel i7-1255U 12th Gen, 1700 Mhz, 10 Cores, 12 LogProc
Lots of RAM: 32 GB
A terrible card for training models: Intel(R) Iris(R) Xe Graphics

So I finished the execution, closed and deleted the Pod, and I am going to run it on my laptop overnight.


In total, I spent 93 cents. I guess I learned a few things. I might try using it again in the future, when my scripts depend more purely on GPUs. For now, Gymnasium seems to use quite a lot of CPU for the step() functions.

To Run the diverBarr pass, use the following command:
- From bin folder run:
    sudo ./opt -load /home/ashwin/Desktop/ipaco/llvm-project/build/lib/LLVMDiver_barr.so -enable-new-pm=0 -cuda-divergence-check divergence_device_ir.ll -o output1.ll

- Output for it:
[ Algorithm ]Processing Function for Barrier Divergence: _Z21barrier_divergent_tidPi
[ Algorithm ] Divergent Barrier Report for _Z21barrier_divergent_tidPi:
[RESULT]   Potential Divergent Barrier Detected at line 0:
  call void @llvm.nvvm.barrier0()
[ Algorithm ] Finished analysis for: _Z21barrier_divergent_tidPi

[ Algorithm ]Processing Function for Barrier Divergence: _Z26barrier_outside_divergencePi
[RESULT]   No divergent barriers detected for _Z26barrier_outside_divergencePi.
[ Algorithm ] Finished analysis for: _Z26barrier_outside_divergencePi

[ Algorithm ]Processing Function for Barrier Divergence: _Z28barrier_NotDivergent_allThdsPi
[RESULT]   No divergent barriers detected for _Z28barrier_NotDivergent_allThdsPi.
[ Algorithm ] Finished analysis for: _Z28barrier_NotDivergent_allThdsPi
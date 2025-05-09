To Run the dataRace pass, use the following command:
- From bin folder run:
    sudo ./opt -load /home/ashwin/Desktop/ipaco/llvm-project/build/lib/LLVMCuBug.so -enable-new-pm=0 -datarace datarace_ir.ll -o output1.ll
    
- Output for it:
[ Algorithm ] Processing Function: _Z11test_case_1Pi
[ Algorithm ]   Memory model size: 5
[ Algorithm ]   Starting Intra-Event Race Checks (Refined)...
_Z11test_case_1Pi:0: Potential Intra-Event Data Race Detected:
  store i32 %3, i32* %5, align 4
_Z11test_case_1Pi:0: Potential Intra-Event Data Race Detected:
  store i32 %6, i32* %8, align 4
[ Algorithm ]   Starting Inter-Event Race Checks...
[ Algorithm ] Finished analysis for: _Z11test_case_1Pi
[ Algorithm ] Processing Function: _Z11test_case_2Pi
[ Algorithm ]   Memory model size: 5
[ Algorithm ]   Starting Intra-Event Race Checks (Refined)...
_Z11test_case_2Pi:0: Potential Intra-Event Data Race Detected:
  store i32 %3, i32* %5, align 4
_Z11test_case_2Pi:0: Potential Intra-Event Data Race Detected:
  store i32 %6, i32* %8, align 4
[ Algorithm ]   Starting Inter-Event Race Checks...
_Z11test_case_2Pi:0&0: Potential Inter-Event Data Race Detected between instructions:
  store i32 %3, i32* %5, align 4
  store i32 %6, i32* %8, align 4
[ Algorithm ] Finished analysis for: _Z11test_case_2Pi
[ Algorithm ] Processing Function: _Z11test_case_3Pi
[ Algorithm ]   Memory model size: 5
[ Algorithm ]   Starting Intra-Event Race Checks (Refined)...
[ Algorithm ]   Starting Inter-Event Race Checks...
[ Algorithm ] Finished analysis for: _Z11test_case_3Pi
To Run the redu_barr pass, use the following command:
- From bin folder run:
    sudo ./opt -load /home/ashwin/Desktop/ipaco/llvm-project/build/lib/LLVMRedu_barr.so -enable-new-pm=0 -redu-barr-check redu_barr.ll -o output1.ll

- Output for it:
[ Algorithm ] Processing Function: _Z11test_case_1Pi
[ Algorithm ]   Memory model size: 5
_Z11test_case_1Pi: Barrier 0 has no inter data race, hence redundant!!
[ Algorithm ] Finished analysis for: _Z11test_case_1Pi
[ Algorithm ] Processing Function: _Z11test_case_2Pi
[ Algorithm ]   Memory model size: 5
_Z11test_case_2Pi:0&0: Potential Inter-Event Data Race Detected between instructions:
  store i32 %3, i32* %7, align 4
  store i32 %8, i32* %13, align 4
[ Algorithm ] Finished analysis for: _Z11test_case_2Pi
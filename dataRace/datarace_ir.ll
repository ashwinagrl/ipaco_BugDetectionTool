; ModuleID = 'datarace.cu'
source_filename = "datarace.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_threadIdx_t = type { i8 }

@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local void @_Z11test_case_1Pi(i32* noundef %0) #0 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %4 = load i32*, i32** %2, align 8
  %5 = getelementptr inbounds i32, i32* %4, i64 0
  store i32 %3, i32* %5, align 4
  call void @llvm.nvvm.barrier0()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %7 = load i32*, i32** %2, align 8
  %8 = getelementptr inbounds i32, i32* %7, i64 0
  store i32 %6, i32* %8, align 4
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local void @_Z11test_case_2Pi(i32* noundef %0) #0 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %4 = load i32*, i32** %2, align 8
  %5 = getelementptr inbounds i32, i32* %4, i64 0
  store i32 %3, i32* %5, align 4
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %7 = load i32*, i32** %2, align 8
  %8 = getelementptr inbounds i32, i32* %7, i64 0
  store i32 %6, i32* %8, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local void @_Z11test_case_3Pi(i32* noundef %0) #0 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %4 = load i32*, i32** %2, align 8
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %6 = zext i32 %5 to i64
  %7 = getelementptr inbounds i32, i32* %4, i64 %6
  store i32 %3, i32* %7, align 4
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %9 = load i32*, i32** %2, align 8
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %11 = zext i32 %10 to i64
  %12 = getelementptr inbounds i32, i32* %9, i64 %11
  store i32 %8, i32* %12, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local void @_Z11test_case_4Pi(i32* noundef %0) #0 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %4 = load i32*, i32** %2, align 8
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %6 = zext i32 %5 to i64
  %7 = getelementptr inbounds i32, i32* %4, i64 %6
  store i32 %3, i32* %7, align 4
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %9 = load i32*, i32** %2, align 8
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %11 = add i32 %10, 1
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds i32, i32* %9, i64 %12
  store i32 %8, i32* %13, align 4
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx75,+sm_75" }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5, !6, !7}
!llvm.ident = !{!8, !9}
!nvvmir.version = !{!10}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{void (i32*)* @_Z11test_case_1Pi, !"kernel", i32 1}
!5 = !{void (i32*)* @_Z11test_case_2Pi, !"kernel", i32 1}
!6 = !{void (i32*)* @_Z11test_case_3Pi, !"kernel", i32 1}
!7 = !{void (i32*)* @_Z11test_case_4Pi, !"kernel", i32 1}
!8 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!9 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!10 = !{i32 2, i32 0}

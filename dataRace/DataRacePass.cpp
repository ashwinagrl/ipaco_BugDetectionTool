#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"     // For instructions()
#include "llvm/Support/raw_ostream.h" // For outs()
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/ValueTracking.h" // For getUnderlyingObject
#include "llvm/IR/DebugInfoMetadata.h"   // For DebugLoc
#include "llvm/IR/Metadata.h"          // For MDNode, MDString, ValueAsMetadata, NamedMDNode
#include "llvm/IR/DataLayout.h"        // For DataLayout
#include <vector>
#include <string>
#include <optional>

using namespace llvm;

// --- Helper Structures (Simplified for Demo) ---
enum class SymbolicLocationType {
    CONSTANT_INDEXED_ARRAY0, // 0
    TID_X_INDEXED_ARRAY,     // 1
    OTHER_COMPLEX,           // 2
    UNINITIALIZED            // 3
};


struct SymbolicLocation {
    const Argument* baseKernelArg = nullptr;
    SymbolicLocationType type = SymbolicLocationType::UNINITIALIZED;

    std::pair<const Argument*, int64_t> getConcreteLocation(int symbolic_tid_x_val) const {
        if (!baseKernelArg) return {nullptr, -1};
        if (type == SymbolicLocationType::CONSTANT_INDEXED_ARRAY0) {
            return {baseKernelArg, 0};
        } else if (type == SymbolicLocationType::TID_X_INDEXED_ARRAY) {
            return {baseKernelArg, static_cast<int64_t>(symbolic_tid_x_val)};
        }
        return {baseKernelArg, -2}; // OTHER_COMPLEX or UNINITIALIZED
    }

    bool operator==(const SymbolicLocation& other) const {
        return baseKernelArg == other.baseKernelArg && type == other.type;
    }

    // Helper to print type
    std::string typeToString() const {
         switch(type) {
            case SymbolicLocationType::CONSTANT_INDEXED_ARRAY0: return "CONSTANT_INDEXED_ARRAY0";
            case SymbolicLocationType::TID_X_INDEXED_ARRAY:     return "TID_X_INDEXED_ARRAY";
            case SymbolicLocationType::OTHER_COMPLEX:           return "OTHER_COMPLEX";
            case SymbolicLocationType::UNINITIALIZED:           return "UNINITIALIZED";
            default: return "UNKNOWN_TYPE";
         }
    }
};

struct AbstractAddressFunc {
    Value* pointerOperandIR = nullptr;
    Function* currentFunction = nullptr;

    AbstractAddressFunc(Value* ptrOp, Function* F) : pointerOperandIR(ptrOp), currentFunction(F) {}

    SymbolicLocation characterize() const {
        SymbolicLocation symLoc;
        symLoc.type = SymbolicLocationType::OTHER_COMPLEX; // Default

        // outs() << "[DEBUG][characterize] Start characterization for pointer: ";
        // if(pointerOperandIR) pointerOperandIR->printAsOperand(outs(), false); else outs() << "null";
        // outs() << "\n";

        if (!pointerOperandIR || !currentFunction || currentFunction->arg_empty()) {
            // outs() << "[DEBUG][characterize]  -> Exiting early (null pointer/func/no args).\n";
            return symLoc;
        }

        symLoc.baseKernelArg = currentFunction->arg_begin();
        // outs() << "[DEBUG][characterize]  -> BaseKernelArg (%0): ";
        // if(symLoc.baseKernelArg) symLoc.baseKernelArg->printAsOperand(outs(), false); else outs() << "null";
        // outs() << "\n";

        GetElementPtrInst* GEP = dyn_cast<GetElementPtrInst>(pointerOperandIR);
        if (!GEP) {
            //  outs() << "[DEBUG][characterize]  -> Pointer is not a GEP.\n";
             return symLoc;
        }

        Value* gepBasePointerValue = GEP->getPointerOperand();
        // outs() << "[DEBUG][characterize]  -> GEP Base Operand: ";
        // if(gepBasePointerValue) gepBasePointerValue->printAsOperand(outs(), false); else outs() << "null";
        // outs() << "\n";

        // --- Revised Check for Base Pointer Origin ---
        bool baseIsFromKernelArg0 = false;
        if (LoadInst* LI = dyn_cast<LoadInst>(gepBasePointerValue)) {
            // outs() << "[DEBUG][characterize]    -> GEP base is a LoadInst.\n";
            Value* loadPtrOp = LI->getPointerOperand(); // This should be the AllocaInst (%2)
            if (AllocaInst* AI = dyn_cast<AllocaInst>(loadPtrOp)) {
                //  outs() << "[DEBUG][characterize]      -> Load is from Alloca: "; AI->printAsOperand(outs(),false); outs() << "\n";
                 // Look for a store to this alloca using the kernel argument
                 // This requires searching through users of the alloca, which is more complex.
                 // Simple approach for this specific IR: Assume if it loads from an alloca,
                 // and that alloca *might* hold the kernel arg, it's our target.
                 // Let's refine: Check if the only store to AI uses F.arg_begin()
                 StoreInst* uniqueStore = nullptr;
                 int storeCount = 0;
                 for (User *U : AI->users()) {
                     if (StoreInst* SI = dyn_cast<StoreInst>(U)) {
                         // Check if this store uses the alloca as the POINTER operand
                         if (SI->getPointerOperand() == AI) {
                             uniqueStore = SI;
                             storeCount++;
                         }
                     }
                 }
                 if (storeCount == 1 && uniqueStore && uniqueStore->getValueOperand() == symLoc.baseKernelArg) {
                    //   outs() << "[DEBUG][characterize]        -> Found unique store to Alloca using Kernel Arg0.\n";
                      baseIsFromKernelArg0 = true;
                 } else {
                    //  outs() << "[DEBUG][characterize]        -> Did not find unique store to Alloca from Kernel Arg0 (Stores found: " << storeCount << ").\n";
                 }
            } else {
                //  outs() << "[DEBUG][characterize]      -> Load is not from an AllocaInst.\n";
            }
        } else {
            // outs() << "[DEBUG][characterize]    -> GEP base is not a LoadInst.\n";
            // Add direct check in case the arg is used directly in GEP (unlikely after standard compilation)
            if (gepBasePointerValue == symLoc.baseKernelArg) {
                //  outs() << "[DEBUG][characterize]    -> GEP base IS directly Kernel Arg0.\n";
                 baseIsFromKernelArg0 = true;
            }
        }
        // --- End Revised Check ---


        if (baseIsFromKernelArg0) { // Use the result of our pattern match
            // outs() << "[DEBUG][characterize]    -> Base derived from kernel arg confirmed.\n";
            if (GEP->getNumIndices() == 1) {
                // outs() << "[DEBUG][characterize]    -> GEP has 1 index.\n";
                Value* indexOperand = GEP->getOperand(1);
                // outs() << "[DEBUG][characterize]      Index Operand: "; indexOperand->print(outs()); outs() << "\n";

                if (ConstantInt* CI = dyn_cast<ConstantInt>(indexOperand)) {
                    // outs() << "[DEBUG][characterize]        Index is ConstantInt: " << CI->getZExtValue() << "\n";
                    if (CI->getZExtValue() == 0) {
                        symLoc.type = SymbolicLocationType::CONSTANT_INDEXED_ARRAY0;
                        // outs() << "[DEBUG][characterize]        Set type = CONSTANT_INDEXED_ARRAY0\n";
                    } else {
                        //  outs() << "[DEBUG][characterize]        Index is non-zero constant. Type = OTHER_COMPLEX\n";
                    }
                } else if (ZExtInst* ZI = dyn_cast<ZExtInst>(indexOperand)) {
                    // outs() << "[DEBUG][characterize]        Index is ZExt: "; ZI->print(outs()); outs() << "\n";
                     if (CallInst* PossibleTidCall = dyn_cast<CallInst>(ZI->getOperand(0))) {
                        // outs() << "[DEBUG][characterize]          ZExt is from Call: "; PossibleTidCall->print(outs()); outs() << "\n";
                         if (Function* Callee = PossibleTidCall->getCalledFunction()) {
                            // outs() << "[DEBUG][characterize]            Callee Name: " << Callee->getName() << "\n";
                             if (Callee->isIntrinsic() && Callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
                                 symLoc.type = SymbolicLocationType::TID_X_INDEXED_ARRAY;
                                //  outs() << "[DEBUG][characterize]            Set type = TID_X_INDEXED_ARRAY\n";
                             } else { //outs() << "[DEBUG][characterize]            Call is not tid.x intrinsic. Type=OTHER_COMPLEX\n"; 
                            }
                         } else { //outs() << "[DEBUG][characterize]          ZExt Call has no Callee? Type=OTHER_COMPLEX\n"; 
                        }
                     } else { //outs() << "[DEBUG][characterize]        ZExt is not from CallInst. Type=OTHER_COMPLEX\n"; 
                    }
                } else {
                    // outs() << "[DEBUG][characterize]      Index is not ConstantInt or ZExt. Type=OTHER_COMPLEX\n";
                 }
            } else {
                //  outs() << "[DEBUG][characterize]    -> GEP has != 1 index. Type=OTHER_COMPLEX\n";
            }
        } else {
             //outs() << "[DEBUG][characterize]    -> Base is NOT derived from kernel arg. Type = OTHER_COMPLEX\n";
        }

        // outs() << "[DEBUG][characterize]  -> Final type for this address: " << symLoc.typeToString() << "\n";
        return symLoc;
    }

    bool isSingleConstantLocationOverTau(const SmallSet<int, 2>& tau) const {
        SymbolicLocation symLoc = characterize();
        if (symLoc.type == SymbolicLocationType::CONSTANT_INDEXED_ARRAY0) return true;
        if (symLoc.type == SymbolicLocationType::TID_X_INDEXED_ARRAY) return tau.size() <= 1;
        return false;
    }
};

enum class AccessMode { READ, WRITE };
struct MemoryAccessEvent {
    int barrierOrder; SmallSet<int, 2> tau; AccessMode mode;
    AbstractAddressFunc addressFunc; Instruction* llvmInstruction; unsigned DbgLine = 0;
    MemoryAccessEvent(int v, const SmallSet<int, 2>& t, AccessMode m, AbstractAddressFunc af, Instruction* I)
        : barrierOrder(v), tau(t), mode(m), addressFunc(std::move(af)), llvmInstruction(I) {
        if (const DebugLoc &DL = I->getDebugLoc()) DbgLine = DL.getLine();
    }
};

// --- The LLVM Pass ---
struct CUDARaceDetectorPass : public FunctionPass {
    static char ID;
    CUDARaceDetectorPass() : FunctionPass(ID) {}

    const int N_THREADS = 2;

    bool runOnFunction(Function &F) override {
        outs() << "[PASS] Processing Function: " << F.getName() << "\n";
        bool isKernel = false;

        CallingConv::ID cc = F.getCallingConv();
        // outs() << "[PASS]   Function CC ID: " << cc << " (PTX_Kernel is " << CallingConv::PTX_Kernel << ")\n";
        if (cc == CallingConv::PTX_Kernel) {
            isKernel = true;
        }

        if (!isKernel) {
            // outs() << "[PASS]   Not PTX_Kernel by CC, checking module-level metadata 'nvvm.annotations'...\n";
            Module *M = F.getParent();
            if (M) {
                NamedMDNode *NMD = M->getNamedMetadata("nvvm.annotations");
                if (NMD) {
                    // outs() << "[PASS]   Found module-level 'nvvm.annotations' NamedMDNode with " << NMD->getNumOperands() << " entries.\n";
                    for (unsigned i = 0, e = NMD->getNumOperands(); i < e; ++i) {
                        MDNode *Annotation = NMD->getOperand(i);
                        if (Annotation && Annotation->getNumOperands() >= 3) {
                             Metadata* funcMd = Annotation->getOperand(0).get();
                             if (ValueAsMetadata *VMD = dyn_cast<ValueAsMetadata>(funcMd)) {
                                 if (Function* AnnFunc = dyn_cast<Function>(VMD->getValue())) {
                                     if (AnnFunc == &F) {
                                         // outs() << "[PASS]     Annotation Entry #" << i << " is for current function: " << F.getName() << ".\n";
                                         Metadata* typeMd = Annotation->getOperand(1).get();
                                         if (MDString *TypeStr = dyn_cast<MDString>(typeMd)) {
                                             // outs() << "[PASS]       MDString value: \"" << TypeStr->getString() << "\"\n";
                                             if (TypeStr->getString() == "kernel") {
                                                //  outs() << "[PASS]     Identified as kernel via metadata.\n";
                                                 isKernel = true;
                                                 break;
                                             }
                                         }
                                     }
                                 }
                             }
                        }
                    }
                } // else { /* outs() << "[PASS]   No module-level 'nvvm.annotations' NamedMDNode found.\n"; */ }
            } // else { /* outs() << "[PASS]   Could not get parent Module.\n"; */ }
        }

        if (!isKernel) {
            // outs() << "[PASS] Function " << F.getName() << " is NOT identified as a kernel. Skipping analysis.\n";
            return false;
        }

        // outs() << "[PASS] Analyzing CUDA Kernel (Confirmed): " << F.getName() << "\n";

        std::vector<MemoryAccessEvent> memoryModel;
        int currentBarrierOrder = 0;
        SmallSet<int, 2> currentTau;
        for (int i = 0; i < N_THREADS; ++i) currentTau.insert(i);

        for (Instruction &I : instructions(F)) {
            if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
                Value* ptrOperand = nullptr;
                AccessMode mode;
                if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
                    ptrOperand = LI->getPointerOperand();
                    mode = AccessMode::READ;
                } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
                    ptrOperand = SI->getPointerOperand();
                    mode = AccessMode::WRITE;
                }
                AbstractAddressFunc af(ptrOperand, &F);
                memoryModel.emplace_back(currentBarrierOrder, currentTau, mode, af, &I);
            } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
                Function *calledFunc = CI->getCalledFunction();
                if (calledFunc && calledFunc->isIntrinsic() &&
                    (calledFunc->getName() == "llvm.nvvm.barrier0")) {
                    currentBarrierOrder++;
                }
            }
        }

        outs() << "[PASS]   Memory model size: " << memoryModel.size() << "\n";

        // ---- Apply Refined Intra-Event Race Rule ----
        outs() << "[PASS]   Starting Intra-Event Race Checks (Refined)...\n"; // DEBUG
        for (size_t i = 0; i < memoryModel.size(); ++i) {
            const auto& event = memoryModel[i];
            // outs() << "[DEBUG][Intra] Checking event #" << i << " (line " << event.DbgLine << ")\n";

            if (event.mode == AccessMode::WRITE && event.tau.size() > 1) {
                // outs() << "[DEBUG][Intra]   Event is WRITE and |tau| > 1.\n";
                SymbolicLocation symLocForEvent = event.addressFunc.characterize();
                // outs() << "[DEBUG][Intra]     Characterized type: " << symLocForEvent.typeToString() << "\n";
                bool collisionFoundForSomePair = false;

                // Refined rule check: Does a collision exist? (i.e. evaluate(t1)==evaluate(t2) for some t1!=t2)
                // For our simplified types:
                if (symLocForEvent.type == SymbolicLocationType::CONSTANT_INDEXED_ARRAY0) {
                    collisionFoundForSomePair = true; // All threads collide at Array[0]
                    // outs() << "[DEBUG][Intra]     Collision found (CONSTANT_INDEXED_ARRAY0 type implies collision for |tau|>1).\n";
                } else if (symLocForEvent.type == SymbolicLocationType::TID_X_INDEXED_ARRAY) {
                    collisionFoundForSomePair = false; // Array[0] != Array[1] for N=2
                    // outs() << "[DEBUG][Intra]     No collision expected (TID_X_INDEXED_ARRAY type).\n";
                } else {
                    //  outs() << "[DEBUG][Intra]     Assuming no collision for OTHER_COMPLEX/UNINITIALIZED types.\n";
                }


                if (collisionFoundForSomePair) {
                    // Refined rule check: Report race UNLESS it maps ALL threads in tau to that same single location.
                    bool isSingleConstLoc = event.addressFunc.isSingleConstantLocationOverTau(event.tau);
                    // outs() << "[DEBUG][Intra]     Is single constant location over tau? " << (isSingleConstLoc ? "Yes" : "No") << "\n";
                    if (!isSingleConstLoc) {
                        // This is where Array[tid%2] would report a race
                        // outs() << "[DEBUG][Intra]     REPORTING INTRA-EVENT RACE!\n";
                        outs() << F.getName() << ":" << event.DbgLine << ": Potential Intra-Event Data Race Detected:\n";
                        event.llvmInstruction->print(outs()); 
                        outs() << "\n";
                    } else {
                        // outs() << "[DEBUG][Intra]     Not reporting race (all threads target same const location).\n";
                    }
                } else {
                    //  outs() << "[DEBUG][Intra]   No collision found between different threads for this event.\n";
                }
            } else {
                //  outs() << "[DEBUG][Intra]   Skipping (Not a Write or |tau| <= 1).\n";
            }
        } // End Intra-Event Loop


        // ---- Apply Inter-Event Race Rule ----
        outs() << "[PASS]   Starting Inter-Event Race Checks...\n";
        for (size_t i = 0; i < memoryModel.size(); ++i) {
            for (size_t j = i + 1; j < memoryModel.size(); ++j) {
                const auto& e_i = memoryModel[i];
                const auto& e_j = memoryModel[j];

                // outs() << "[DEBUG][Inter] Comparing event #" << i << " (v=" << e_i.barrierOrder << ") with event #" << j << " (v=" << e_j.barrierOrder << ")\n"; // Verbose

                if (e_i.barrierOrder != e_j.barrierOrder) {
                    continue; // Different barrier orders
                }
                if (!(e_i.mode == AccessMode::WRITE || e_j.mode == AccessMode::WRITE)) {
                    continue; // Neither is a write
                }

                // outs() << "[DEBUG][Inter]   Checking clash for pair (" << i << "," << j << ") in barrier " << e_i.barrierOrder << "...\n";

                bool interClash = false;
                SymbolicLocation symLoc_i = e_i.addressFunc.characterize();
                SymbolicLocation symLoc_j = e_j.addressFunc.characterize();

                // outs() << "[DEBUG][Inter]     Event " << i << " symLoc type: " << symLoc_i.typeToString() << "\n";
                // outs() << "[DEBUG][Inter]     Event " << j << " symLoc type: " << symLoc_j.typeToString() << "\n";

                if (symLoc_i.type != SymbolicLocationType::UNINITIALIZED && symLoc_i.type != SymbolicLocationType::OTHER_COMPLEX &&
                    symLoc_j.type != SymbolicLocationType::UNINITIALIZED && symLoc_j.type != SymbolicLocationType::OTHER_COMPLEX) {

                    // outs() << "[DEBUG][Inter]       Checking concrete locations (N=2)...\n";
                    std::pair<const Argument*, int64_t> concreteLoc_i_t0 = symLoc_i.getConcreteLocation(0);
                    std::pair<const Argument*, int64_t> concreteLoc_j_t1 = symLoc_j.getConcreteLocation(1);
                    // outs() << "[DEBUG][Inter]         e_i (tid 0): Base=" << (void*)concreteLoc_i_t0.first << ", Offset=" << concreteLoc_i_t0.second << "\n";
                    // outs() << "[DEBUG][Inter]         e_j (tid 1): Base=" << (void*)concreteLoc_j_t1.first << ", Offset=" << concreteLoc_j_t1.second << "\n";

                    if (concreteLoc_i_t0.first != nullptr && concreteLoc_i_t0 == concreteLoc_j_t1) {
                        interClash = true;
                        // outs() << "[DEBUG][Inter]         CLASH FOUND (i0 vs j1)!\n";
                    }

                    if (!interClash) {
                        std::pair<const Argument*, int64_t> concreteLoc_i_t1 = symLoc_i.getConcreteLocation(1);
                        std::pair<const Argument*, int64_t> concreteLoc_j_t0 = symLoc_j.getConcreteLocation(0);
                        // outs() << "[DEBUG][Inter]         e_i (tid 1): Base=" << (void*)concreteLoc_i_t1.first << ", Offset=" << concreteLoc_i_t1.second << "\n";
                        // outs() << "[DEBUG][Inter]         e_j (tid 0): Base=" << (void*)concreteLoc_j_t0.first << ", Offset=" << concreteLoc_j_t0.second << "\n";

                        if (concreteLoc_i_t1.first != nullptr && concreteLoc_i_t1 == concreteLoc_j_t0) {
                            interClash = true;
                            // outs() << "[DEBUG][Inter]         CLASH FOUND (i1 vs j0)!\n";
                        }
                    }
                } else {
                    // outs() << "[DEBUG][Inter]     Skipping concrete comparison: One or both types complex/uninitialized.\n";
                }

                if (interClash) {
                    outs() << F.getName() << ":" << e_i.DbgLine << "&" << e_j.DbgLine
                           << ": Potential Inter-Event Data Race Detected between instructions:\n";
                    e_i.llvmInstruction->print(outs()); outs() << "\n";
                    e_j.llvmInstruction->print(outs()); outs() << "\n";
                } // else { /* outs() << "[DEBUG][Inter]   No inter-clash found for pair (" << i << "," << j << ").\n"; */ }
            }
        }
        outs() << "[PASS] Finished analysis for: " << F.getName() << "\n";
        return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();
    }
};

char CUDARaceDetectorPass::ID = 0;
static RegisterPass<CUDARaceDetectorPass> X(
    "datarace", // Your chosen name
    "CUDA Data Race Detector Pass (Tailored with Full Debugging)",
    false,
    false);
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"     // For instructions()
#include "llvm/Support/raw_ostream.h" // For outs()
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/ValueTracking.h" // For getUnderlyingObject, GetPointerBaseWithConstantOffset
#include "llvm/IR/DebugInfoMetadata.h"   // For DebugLoc
#include "llvm/IR/Metadata.h"          // For MDNode, MDString, ValueAsMetadata, NamedMDNode
#include "llvm/IR/DataLayout.h"        // For DataLayout
#include "llvm/IR/GlobalVariable.h"    // For checking shared memory globals
#include <vector>
#include <string>
#include <optional>
#include <map>
#include <tuple> // Include tuple header

using namespace llvm;

// --- Helper Structures (Simplified for Demo) ---
enum class MemSpace { GLOBAL, SHARED, OTHER };

enum class SymbolicLocationType {
    CONSTANT_OFFSET,  // Base[Const], includes Array[0], shared_val
    TID_X_INDEXED,     // Base[tid.x], includes Array[tid], smem[tid]
    // TID_X_OFFSETTED, // Base[tid.x +/- Const] -> Treat as OTHER_COMPLEX for now
    OTHER_COMPLEX,
    UNINITIALIZED
};

struct SymbolicLocation {
    const Value* baseObject = nullptr; // Can be Argument* or GlobalVariable*
    MemSpace memSpace = MemSpace::OTHER;
    SymbolicLocationType type = SymbolicLocationType::UNINITIALIZED;
    int64_t constantOffsetIfAny = 0; // Valid if type is CONSTANT_OFFSET

    std::tuple<const Value*, int64_t, MemSpace> getConcreteLocation(int symbolic_tid_x_val) const {
        if (!baseObject) return {nullptr, -1, MemSpace::OTHER};
        int64_t offset = -2;
        if (type == SymbolicLocationType::CONSTANT_OFFSET) {
            offset = constantOffsetIfAny;
        } else if (type == SymbolicLocationType::TID_X_INDEXED) {
            offset = static_cast<int64_t>(symbolic_tid_x_val);
        }
        return {baseObject, offset, memSpace};
    }

    bool operator==(const SymbolicLocation& other) const {
        return baseObject == other.baseObject && memSpace == other.memSpace && type == other.type;
    }

    std::string typeToString() const {
         switch(type) {
            case SymbolicLocationType::CONSTANT_OFFSET:  return "CONSTANT_OFFSET";
            case SymbolicLocationType::TID_X_INDEXED:     return "TID_X_INDEXED";
            case SymbolicLocationType::OTHER_COMPLEX:    return "OTHER_COMPLEX";
            case SymbolicLocationType::UNINITIALIZED:    return "UNINITIALIZED";
            default: return "UNKNOWN_TYPE (" + std::to_string(static_cast<int>(type)) + ")";
         }
    }
    std::string spaceToString() const {
        switch(memSpace) {
            case MemSpace::GLOBAL: return "GLOBAL";
            case MemSpace::SHARED: return "SHARED";
            case MemSpace::OTHER:  return "OTHER";
            default: return "UNKNOWN_SPACE";
        }
    }
};

// Updated AbstractAddressFunc using GetPointerBaseWithConstantOffset
struct AbstractAddressFunc {
    Value* pointerOperandIR = nullptr;
    Function* currentFunction = nullptr;
    const DataLayout& DL; // Store DataLayout

    AbstractAddressFunc(Value* ptrOp, Function* F) :
        pointerOperandIR(ptrOp), currentFunction(F), DL(F->getParent()->getDataLayout()) {}

    SymbolicLocation characterize() const {
        SymbolicLocation symLoc;
        symLoc.type = SymbolicLocationType::OTHER_COMPLEX; // Default
        symLoc.memSpace = MemSpace::OTHER;

        outs() << "[DEBUG][characterize] Start characterization for pointer: ";
        if(pointerOperandIR) { pointerOperandIR->print(outs()); } else { outs() << "null"; }
        outs() << "\n";

        if (!pointerOperandIR || !currentFunction) {
            outs() << "[DEBUG][characterize]  -> Exiting early (null pointer/func).\n";
            return symLoc;
        }

        int64_t constOffset = 0;
        // Use GetPointerBaseWithConstantOffset to find the base and const offset
        Value* baseValue = llvm::GetPointerBaseWithConstantOffset(pointerOperandIR, constOffset, DL);

        outs() << "[DEBUG][characterize]  -> GetPointerBase result: Base=";
        if(baseValue) baseValue->printAsOperand(outs(), false); else outs() << "null";
        outs() << ", ConstOffset=" << constOffset << "\n";

        if (!baseValue) {
             outs() << "[DEBUG][characterize]  -> Could not find base via GetPointerBase... Type=OTHER_COMPLEX\n";
             return symLoc; // Couldn't resolve simply
        }

        // Identify Memory Space and Base Object from the resolved baseValue
        if (const Argument* Arg = dyn_cast<Argument>(baseValue)) {
            if (Arg->getParent() == currentFunction) {
                symLoc.memSpace = MemSpace::GLOBAL;
                symLoc.baseObject = Arg;
                 outs() << "[DEBUG][characterize]    -> Base identified as Kernel Argument: "; Arg->printAsOperand(outs(), false); outs() << "\n";
            }
        } else if (const GlobalVariable* GV = dyn_cast<GlobalVariable>(baseValue)) {
             outs() << "[DEBUG][characterize]    -> Base is GlobalVariable. AddrSpace: " << GV->getAddressSpace() << "\n";
             if (GV->getAddressSpace() == 3) {
                symLoc.memSpace = MemSpace::SHARED;
                symLoc.baseObject = GV;
                outs() << "[DEBUG][characterize]    -> Base identified as Shared GlobalVar: "; GV->printAsOperand(outs(), false); outs() << "\n";
             }
        }

        if (symLoc.memSpace == MemSpace::OTHER) {
             outs() << "[DEBUG][characterize]  -> Base object not identified as KernelArg or SharedGlobal. Type=OTHER_COMPLEX\n";
             return symLoc;
        }

        // Determine Access Type
        // If GetPointerBase... found the *original* pointerOperandIR as the base, it means
        // the offset wasn't constant or wasn't a simple GEP it could resolve.
        // OR If the baseValue it found *is* the pointerOperandIR (e.g. direct arg use)
        // -> Need to check if pointerOperandIR itself is GEP to look for tid
        // If GetPointerBase returned a *different* base, it implies a constant offset was found.

        if (baseValue != pointerOperandIR) {
             // GetPointerBase resolved through a GEP with constant offset(s)
             symLoc.type = SymbolicLocationType::CONSTANT_OFFSET;
             symLoc.constantOffsetIfAny = constOffset;
             outs() << "[DEBUG][characterize]    -> GetPointerBase found const offset. Set Type=CONSTANT_OFFSET(" << constOffset << ")\n";
        } else {
             // GetPointerBase couldn't resolve offset, baseValue is likely pointerOperandIR.
             // Now check if pointerOperandIR is a GEP involving tid.x
             if (GetElementPtrInst* GEP = dyn_cast<GetElementPtrInst>(pointerOperandIR)) {
                 outs() << "[DEBUG][characterize]    -> GetPointerBase returned original ptr, checking GEP indices...\n";
                 // Analyze GEP indices specifically for tid pattern
                 if (GEP->getNumIndices() == 1 || (GEP->getNumIndices() == 2 && isa<ConstantInt>(GEP->getOperand(1)) && cast<ConstantInt>(GEP->getOperand(1))->isZero())) {
                    Value* indexOperand = GEP->getOperand(GEP->getNumIndices()); // Last index
                    outs() << "[DEBUG][characterize]      -> Last Index Operand Value: "; indexOperand->print(outs()); outs() << "\n";
                    if (CastInst* CastI = dyn_cast<CastInst>(indexOperand)) {
                        outs() << "[DEBUG][characterize]        -> Index is CastInst: "; CastI->print(outs()); outs() << "\n";
                        if (CallInst* PossibleTidCall = dyn_cast<CallInst>(CastI->getOperand(0))) {
                           outs() << "[DEBUG][characterize]          -> Cast is from CallInst: "; PossibleTidCall->print(outs()); outs() << "\n";
                            if (Function* Callee = PossibleTidCall->getCalledFunction()) {
                               outs() << "[DEBUG][characterize]            -> Callee Name: " << Callee->getName() << "\n";
                                if (Callee->isIntrinsic() && Callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
                                    symLoc.type = SymbolicLocationType::TID_X_INDEXED;
                                    outs() << "[DEBUG][characterize]            -> Set type = TID_X_INDEXED\n";
                                }
                            }
                        }
                    } // else type remains OTHER_COMPLEX
                } // else GEP has weird indices, type remains OTHER_COMPLEX
             } else if (symLoc.memSpace != MemSpace::OTHER) {
                 // Direct use of base object (e.g., shared scalar identified via getUnderlyingObject earlier)
                 symLoc.type = SymbolicLocationType::CONSTANT_OFFSET;
                 symLoc.constantOffsetIfAny = 0; // Offset relative to base object itself
                 outs() << "[DEBUG][characterize]    -> Direct use of base object. Set Type=CONSTANT_OFFSET(0)\n";
             }
        } // End check if GetPointerBase resolved offset

        outs() << "[DEBUG][characterize]  -> Final type: " << symLoc.typeToString() << ", Space: " << symLoc.spaceToString() << "\n";
        return symLoc;
    }

    // Checks if address is guaranteed single constant location for all in tau
    bool isSingleConstantLocationOverTau(const SmallSet<int, 2>& tau) const {
        SymbolicLocation symLoc = characterize();
        if (symLoc.type == SymbolicLocationType::CONSTANT_OFFSET) return true;
        if (symLoc.type == SymbolicLocationType::TID_X_INDEXED) return tau.size() <= 1;
        return false;
    }
};

// --- MemoryAccessEvent, BarrierEvent structs remain the same ---
enum class AccessMode { READ, WRITE };
struct MemoryAccessEvent {
    int barrierOrder; SmallSet<int, 2> tau; AccessMode mode;
    AbstractAddressFunc addressFunc; Instruction* llvmInstruction; unsigned DbgLine = 0;
    MemoryAccessEvent(int v, const SmallSet<int, 2>& t, AccessMode m, AbstractAddressFunc af, Instruction* I)
        : barrierOrder(v), tau(t), mode(m), addressFunc(std::move(af)), llvmInstruction(I) {
        if (const DebugLoc &DL = I->getDebugLoc()) DbgLine = DL.getLine();
    }
};
struct BarrierEvent {
    int precedingBarrierOrder; Instruction* barrierInst; unsigned DbgLine = 0;
     BarrierEvent(int k, Instruction* I) : precedingBarrierOrder(k), barrierInst(I) {
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

        // --- Kernel Identification Logic (as corrected before) ---
        CallingConv::ID cc = F.getCallingConv();
        if (cc == CallingConv::PTX_Kernel) { isKernel = true; }
        if (!isKernel) {
            Module *M = F.getParent();
            if (M) {
                NamedMDNode *NMD = M->getNamedMetadata("nvvm.annotations");
                if (NMD) {
                    for (unsigned i = 0, e = NMD->getNumOperands(); i < e; ++i) {
                        MDNode *Annotation = NMD->getOperand(i);
                        if (Annotation && Annotation->getNumOperands() >= 3) {
                             Metadata* funcMd = Annotation->getOperand(0).get();
                             if (ValueAsMetadata *VMD = dyn_cast<ValueAsMetadata>(funcMd)) {
                                 if (Function* AnnFunc = dyn_cast<Function>(VMD->getValue())) {
                                     if (AnnFunc == &F) {
                                         Metadata* typeMd = Annotation->getOperand(1).get();
                                         if (MDString *TypeStr = dyn_cast<MDString>(typeMd)) {
                                             if (TypeStr->getString() == "kernel") {
                                                 outs() << "[PASS]     Identified as kernel via metadata.\n";
                                                 isKernel = true;
                                                 break;
                                             } } } } } } } } }
        }
        // --- End Kernel Identification ---

        if (!isKernel) {
            outs() << "[PASS] Function " << F.getName() << " is NOT identified as a kernel. Skipping analysis.\n";
            return false;
        }

        outs() << "[PASS] Analyzing CUDA Kernel (Confirmed): " << F.getName() << "\n";

        std::vector<MemoryAccessEvent> memoryModel;
        std::vector<BarrierEvent> barriersFound;
        int currentBarrierOrder = 0;
        SmallSet<int, 2> currentTau;
        for (int i = 0; i < N_THREADS; ++i) currentTau.insert(i);

        // Build memory model & record barriers
        for (Instruction &I : instructions(F)) {
            if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
                Value* ptrOperand = nullptr;
                AccessMode mode;
                if (LoadInst *LI = dyn_cast<LoadInst>(&I)) { ptrOperand = LI->getPointerOperand(); mode = AccessMode::READ; }
                else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) { ptrOperand = SI->getPointerOperand(); mode = AccessMode::WRITE; }
                AbstractAddressFunc af(ptrOperand, &F);
                memoryModel.emplace_back(currentBarrierOrder, currentTau, mode, af, &I);
            } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
                Function *calledFunc = CI->getCalledFunction();
                if (calledFunc && calledFunc->isIntrinsic() && (calledFunc->getName() == "llvm.nvvm.barrier0")) {
                    barriersFound.emplace_back(currentBarrierOrder, CI);
                    currentBarrierOrder++;
                }
            }
        }
        int V_max = currentBarrierOrder + 1;

        outs() << "[PASS]   Memory model size: " << memoryModel.size() << "\n";
        outs() << "[PASS]   Barriers found: " << barriersFound.size() << " (Max interval index=" << V_max - 1 << ")\n";



        // ---- Redundant Barrier Detection ----
        outs() << "[PASS]   Starting Redundant Barrier Checks...\n";
        std::map<int, std::vector<const MemoryAccessEvent*>> eventsByOrder;
        for (const auto& event : memoryModel) {
            eventsByOrder[event.barrierOrder].push_back(&event);
        }

        std::set<int> redundantBarrierIntervals;

        for (int k = 0; k < V_max - 1; ++k) {
             outs() << "[DEBUG][Barrier] Checking barrier after interval k=" << k << "\n";
             bool is_barrier_k_necessary = false;
             if (eventsByOrder.find(k) == eventsByOrder.end() || eventsByOrder.find(k+1) == eventsByOrder.end()) {
                 outs() << "[DEBUG][Barrier]   Skipping barrier k=" << k << ": No events found in interval " << k << " or " << k+1 << ".\n";
                 continue; // Skip if one of the intervals is empty
             }
             const auto& events_k = eventsByOrder.at(k);
             const auto& events_k_plus_1 = eventsByOrder.at(k+1);

             outs() << "[DEBUG][Barrier]   Interval " << k << " has " << events_k.size() << " events.\n";
             outs() << "[DEBUG][Barrier]   Interval " << k+1 << " has " << events_k_plus_1.size() << " events.\n";

             for (const MemoryAccessEvent* e_i_ptr : events_k) {
                 for (const MemoryAccessEvent* e_j_ptr : events_k_plus_1) {
                    const auto& e_i = *e_i_ptr;
                    const auto& e_j = *e_j_ptr;

                    outs() << "[DEBUG][Barrier]     Comparing k=" << k << " event (line " << e_i.DbgLine << ") vs k=" << k+1 << " event (line " << e_j.DbgLine << ")\n";
                    outs() << "[DEBUG][Barrier]       "; e_i.llvmInstruction->print(outs()); outs() << "\n";
                    outs() << "[DEBUG][Barrier]       "; e_j.llvmInstruction->print(outs()); outs() << "\n";

                    if (e_i.mode == AccessMode::READ && e_j.mode == AccessMode::READ) {
                         outs() << "[DEBUG][Barrier]       Skipping R-R pair.\n";
                         continue;
                    }

                    bool barrierConflict = false;
                    SymbolicLocation symLoc_i = e_i.addressFunc.characterize();
                    SymbolicLocation symLoc_j = e_j.addressFunc.characterize();

                    outs() << "[DEBUG][Barrier]       Event i type: " << symLoc_i.typeToString() << ", Space: " << symLoc_i.spaceToString() << ", Base: ";
                    if(symLoc_i.baseObject) symLoc_i.baseObject->printAsOperand(outs(), false); else outs() << "null"; outs() << "\n";
                    outs() << "[DEBUG][Barrier]       Event j type: " << symLoc_j.typeToString() << ", Space: " << symLoc_j.spaceToString() << ", Base: ";
                    if(symLoc_j.baseObject) symLoc_j.baseObject->printAsOperand(outs(), false); else outs() << "null"; outs() << "\n";

                    // Alias Check: Different memory space or different base object means no alias
                    if (symLoc_i.memSpace == MemSpace::OTHER || symLoc_j.memSpace == MemSpace::OTHER || !symLoc_i.baseObject || !symLoc_j.baseObject || symLoc_i.memSpace != symLoc_j.memSpace || symLoc_i.baseObject != symLoc_j.baseObject) {
                         if (symLoc_i.baseObject != symLoc_j.baseObject || symLoc_i.memSpace != symLoc_j.memSpace) {
                             outs() << "[DEBUG][Barrier]         No alias (different base object or memory space). Skipping concrete check.\n";
                         } else {
                             outs() << "[DEBUG][Barrier]         Cannot check alias (one base/space is OTHER/null). Skipping concrete check.\n";
                         }
                         continue; // Cannot alias or cannot compare base
                    }

                    // If same base object/space, check for cross-thread collision if types are simple
                    if (symLoc_i.type != SymbolicLocationType::UNINITIALIZED && 
                        symLoc_i.type != SymbolicLocationType::OTHER_COMPLEX &&
                        symLoc_j.type != SymbolicLocationType::UNINITIALIZED && 
                        symLoc_j.type != SymbolicLocationType::OTHER_COMPLEX) {
                    
                        outs() << "[DEBUG][Barrier]         Checking concrete locations (N=2)...\n";
                        std::tuple<const Value*, int64_t, MemSpace> concreteLoc_i_t0 = symLoc_i.getConcreteLocation(0);
                        std::tuple<const Value*, int64_t, MemSpace> concreteLoc_j_t1 = symLoc_j.getConcreteLocation(1);
                        
                        // Access OFFSET (second element, index=1)
                        outs() << "[DEBUG][Barrier]           i(t0): Offset=" << std::get<1>(concreteLoc_i_t0) << "\n";
                        outs() << "[DEBUG][Barrier]           j(t1): Offset=" << std::get<1>(concreteLoc_j_t1) << "\n";
                    
                        // Compare offsets (index=1)
                        if (std::get<1>(concreteLoc_i_t0) >= -1 && 
                            std::get<1>(concreteLoc_j_t1) >= -1 &&
                            std::get<1>(concreteLoc_i_t0) == std::get<1>(concreteLoc_j_t1)) {
                            barrierConflict = true;
                            outs() << "[DEBUG][Barrier]           CONFLICT FOUND (i0 vs j1)!\n";
                        }
                    
                        if (!barrierConflict) {
                            std::tuple<const Value*, int64_t, MemSpace> concreteLoc_i_t1 = symLoc_i.getConcreteLocation(1);
                            std::tuple<const Value*, int64_t, MemSpace> concreteLoc_j_t0 = symLoc_j.getConcreteLocation(0);
                            
                            // Access OFFSET (index=1)
                            outs() << "[DEBUG][Barrier]           i(t1): Offset=" << std::get<1>(concreteLoc_i_t1) << "\n";
                            outs() << "[DEBUG][Barrier]           j(t0): Offset=" << std::get<1>(concreteLoc_j_t0) << "\n";
                    
                            if (std::get<1>(concreteLoc_i_t1) >= -1 && 
                                std::get<1>(concreteLoc_j_t0) >= -1 &&
                                std::get<1>(concreteLoc_i_t1) == std::get<1>(concreteLoc_j_t0)) {
                                barrierConflict = true;
                                outs() << "[DEBUG][Barrier]           CONFLICT FOUND (i1 vs j0)!\n";
                            }
                        }
                    } else {
                         outs() << "[DEBUG][Barrier]         Skipping concrete comparison: One or both types complex/uninitialized.\n";
                         // Conservative decision for OTHER_COMPLEX? Assume necessary?
                         // outs() << "[DEBUG][Barrier]         Conservatively assuming complex types might conflict.\n";
                         // barrierConflict = true; // Uncomment for conservative approach
                    }

                    if (barrierConflict) {
                        outs() << "[DEBUG][Barrier]       Conflict found for barrier " << k << ". Marking as necessary.\n";
                        is_barrier_k_necessary = true;
                        break; // Break inner loop (j)
                    } else {
                        // outs() << "[DEBUG][Barrier]       No conflict found for this pair.\n"; // Verbose
                    }
                 } // end loop j
                 if (is_barrier_k_necessary) break; // Break outer loop (i)
             } // end loop i

            if (!is_barrier_k_necessary) {
                outs() << "[DEBUG][Barrier]   Barrier k=" << k << " determined to be REDUNDANT.\n";
                redundantBarrierIntervals.insert(k);
            } else {
                 outs() << "[DEBUG][Barrier]   Barrier k=" << k << " determined to be NECESSARY.\n";
            }
        } // end loop k

        // Report Redundant Barriers
        if (!redundantBarrierIntervals.empty()) {
             outs() << "[PASS] Redundant Barrier Report for " << F.getName() << ":\n";
             for (const auto& barrierEvent : barriersFound) {
                 if (redundantBarrierIntervals.count(barrierEvent.precedingBarrierOrder)) {
                     outs() << "  Potential Redundant Barrier Detected at line " << barrierEvent.DbgLine << " (following interval " << barrierEvent.precedingBarrierOrder << "):\n";
                     barrierEvent.barrierInst->print(outs()); outs() << "\n";
                 }
             }
        } else {
            outs() << "[PASS] No redundant barriers detected for " << F.getName() << ".\n";
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
    "redu_barr", // Keep user's chosen name
    "CUDA Data Race and Redundant Barrier Detector (Conceptual V5 + Debug)",
    false,
    true); // Analysis Pass
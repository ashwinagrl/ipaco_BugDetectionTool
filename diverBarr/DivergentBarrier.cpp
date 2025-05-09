#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;

template <typename T> class AbstractValue {
public:
  AbstractValue() = default;
  AbstractValue(const AbstractValue &other) = default;
  virtual ~AbstractValue() = 0; // Pure virtual destructor
  virtual std::string getString() const = 0;
  virtual T join(const T &v) const = 0;
};
template <typename T>
AbstractValue<T>::~AbstractValue() {} // Provide definition

template <typename T> class PointerAbstractValue : public AbstractValue<T> {
public:
  PointerAbstractValue() : isAddressType_(false) {}
  PointerAbstractValue(const PointerAbstractValue &other) = default;
  virtual ~PointerAbstractValue()
      override = 0; // Pure virtual destructor override
  bool isAddressType() const { return isAddressType_; }
  void setAddressType() { isAddressType_ = true; }
  void setValueType() { isAddressType_ = false; }
  virtual std::string getString() const override = 0;
  virtual T join(const T &v) const override = 0;

protected:
  bool isAddressType_;
};
template <typename T>
PointerAbstractValue<T>::~PointerAbstractValue() {} // Provide definition

// Forward declare derived classes BEFORE AbstractState
class MultiplierValue;
class GPUState;

// --- AbstractState Definition ---
template <typename T_Val, typename T_State> class AbstractState {
public:
  AbstractState() = default;
  AbstractState(const AbstractState &other) = default;
  virtual ~AbstractState() = default;

  // Pure virtual operators, derived must implement
  virtual bool operator==(const AbstractState &other) const = 0;
  virtual void operator=(const AbstractState &st) = 0;

  bool operator!=(const AbstractState &st) const { return !(*this == st); }
  void clear() { valueMap_.clear(); }
  bool hasValue(const Value *in) const {
    return (valueMap_.find(in) != valueMap_.end());
  }
  virtual T_Val getValue(const Value *in) const = 0;
  virtual void setValue(const Value *in, T_Val v) { valueMap_[in] = v; }
  virtual T_State mergeState(const T_State &st) const = 0;
  virtual std::string getString() const;

protected:
  std::map<const Value *, T_Val> valueMap_;
};

template <typename T_Val, typename T_State>
std::string AbstractState<T_Val, T_State>::getString() const {
  std::string s;
  s.append("[");
  bool first = true;
  for (auto const &map_entry : valueMap_) {
    const Value *val = map_entry.first;
    const T_Val &abstract_val = map_entry.second;
    if (!first) {
      s.append(", ");
    }
    if (val && val->hasName()) {
      s.append(val->getName().str());
    } else if (val) {
      s.append("<tmp_val>");
    } else {
      s.append("<null_val>");
    }
    s.append(":").append(abstract_val.getString());
    first = false;
  }
  s.append("]");
  return s;
}
// --- End AbstractState Definition ---

// --- MultiplierValue Definition ---
enum MultiplierValueType { BOT, ZERO, ONE, NEGONE, TOP };

class MultiplierValue : public PointerAbstractValue<MultiplierValue> {
public:
  MultiplierValueType t_;
  bool isBool_;

  MultiplierValue(MultiplierValueType t = BOT, bool is_bool = false)
      : t_(t), isBool_(is_bool) {}
  ~MultiplierValue() override = default;

  MultiplierValue(const MultiplierValue &other) = default;
  MultiplierValue &operator=(const MultiplierValue &other) = default;

  static MultiplierValue getMultiplierValue(int x, bool b);
  int getIntValue() const;
  MultiplierValue join(const MultiplierValue &v) const override;
  std::string getString() const override;
  MultiplierValueType getType() const { return t_; }

  friend MultiplierValue operator+(const MultiplierValue &v1,
                                   const MultiplierValue &v2);
  friend MultiplierValue operator*(const MultiplierValue &v1,
                                   const MultiplierValue &v2);
  friend MultiplierValue operator-(const MultiplierValue &v);
  friend MultiplierValue operator&&(const MultiplierValue &v1,
                                    const MultiplierValue &v2);
  friend MultiplierValue operator||(const MultiplierValue &v1,
                                    const MultiplierValue &v2);
  friend MultiplierValue eq(const MultiplierValue &v1,
                            const MultiplierValue &v2);
  friend MultiplierValue neq(const MultiplierValue &v1,
                             const MultiplierValue &v2);
  friend bool operator==(const MultiplierValue &v1, const MultiplierValue &v2);
  friend bool operator!=(const MultiplierValue &v1, const MultiplierValue &v2);
};
// --- End MultiplierValue Definition ---

// --- Implementation for MultiplierValue methods and friends ---
// [Implementations MUST be provided here based on previous versions/paper]
MultiplierValue MultiplierValue::getMultiplierValue(int x, bool b) {
  if (x == 0)
    return MultiplierValue(ZERO, b);
  if (x == 1)
    return MultiplierValue(ONE, b);
  if (x == -1)
    return MultiplierValue(NEGONE, b);
  return MultiplierValue(TOP);
}
int MultiplierValue::getIntValue() const {
  if (t_ == ZERO)
    return 0;
  if (t_ == ONE)
    return 1;
  if (t_ == NEGONE)
    return -1;
  return 999;
}
MultiplierValue MultiplierValue::join(const MultiplierValue &v) const {
  if (t_ == BOT)
    return v;
  if (v.t_ == BOT)
    return *this;
  if (t_ == v.t_)
    return v;
  return MultiplierValue(TOP, isBool_ || v.isBool_);
}
std::string MultiplierValue::getString() const {
  std::string s;
  if (isAddressType())
    s.append("*");
  switch (t_) {
  case BOT:
    return s.append("B");
  case ZERO:
    return s.append("0");
  case ONE:
    return s.append("1");
  case NEGONE:
    return s.append("-1");
  case TOP:
  default:
    return s.append("T");
  }
}
MultiplierValue operator+(const MultiplierValue &v1,
                          const MultiplierValue &v2) {
  if (v1.t_ == BOT || v2.t_ == BOT)
    return MultiplierValue(BOT);
  if (v1.t_ == TOP || v2.t_ == TOP)
    return MultiplierValue(TOP);
  int out = v1.getIntValue() + v2.getIntValue();
  return MultiplierValue::getMultiplierValue(out, false);
}
MultiplierValue operator*(const MultiplierValue &v1,
                          const MultiplierValue &v2) {
  if (v1.t_ == BOT || v2.t_ == BOT)
    return MultiplierValue(BOT);
  if (v1.t_ == ZERO || v2.t_ == ZERO)
    return MultiplierValue(ZERO);
  if (v1.t_ == TOP || v2.t_ == TOP)
    return MultiplierValue(TOP);
  int out = v1.getIntValue() * v2.getIntValue();
  return MultiplierValue::getMultiplierValue(out, false);
}
MultiplierValue operator-(const MultiplierValue &v) {
  if (v.t_ == BOT)
    return MultiplierValue(BOT);
  if (v.t_ == TOP)
    return MultiplierValue(TOP);
  if (v.isBool_) {
    if (v.t_ == ZERO)
      return MultiplierValue(NEGONE, true);
    if (v.t_ == ONE || v.t_ == NEGONE)
      return MultiplierValue(ZERO, true);
  }
  int out = -v.getIntValue();
  return MultiplierValue::getMultiplierValue(out, v.isBool_);
}
MultiplierValue operator&&(const MultiplierValue &v1,
                           const MultiplierValue &v2) {
  if (v1.t_ == BOT || v2.t_ == BOT)
    return MultiplierValue(BOT);
  if (v1.t_ == ZERO || v2.t_ == ZERO)
    return MultiplierValue(ZERO, true);
  if (v1.t_ == ONE || v2.t_ == ONE)
    return MultiplierValue(ONE, true);
  return MultiplierValue(TOP);
}
MultiplierValue operator||(const MultiplierValue &v1,
                           const MultiplierValue &v2) {
  if (v1.t_ == BOT || v2.t_ == BOT)
    return MultiplierValue(BOT);
  if (v1.t_ == NEGONE || v2.t_ == NEGONE)
    return MultiplierValue(NEGONE, true);
  if (v1.t_ == ZERO && v2.t_ == ZERO)
    return MultiplierValue(ZERO, true);
  if ((v1.t_ == ONE && v2.t_ == ZERO) || (v1.t_ == ZERO && v2.t_ == ONE))
    return MultiplierValue(ONE, true);
  return MultiplierValue(TOP);
}
MultiplierValue eq(const MultiplierValue &v1, const MultiplierValue &v2) {
  if (v1.t_ == BOT || v2.t_ == BOT)
    return MultiplierValue(BOT);
  if (v1.t_ == v2.t_ && v1.t_ != TOP)
    return MultiplierValue(ZERO, true);
  if (((v1.t_ == ONE || v1.t_ == NEGONE) && v2.t_ == ZERO) ||
      ((v2.t_ == ONE || v2.t_ == NEGONE) && v1.t_ == ZERO)) {
    return MultiplierValue(ONE, true);
  }
  return MultiplierValue(TOP);
}
MultiplierValue neq(const MultiplierValue &v1, const MultiplierValue &v2) {
  if (v1.t_ == BOT || v2.t_ == BOT)
    return MultiplierValue(BOT);
  if (v1.t_ == v2.t_ && v1.t_ != TOP)
    return MultiplierValue(ZERO, true);
  if (((v1.t_ == ONE || v1.t_ == NEGONE) && v2.t_ == ZERO) ||
      ((v2.t_ == ONE || v2.t_ == NEGONE) && v1.t_ == ZERO)) {
    return MultiplierValue(NEGONE, true);
  }
  return MultiplierValue(TOP);
}
bool operator==(const MultiplierValue &v1, const MultiplierValue &v2) {
  return (v1.t_ == v2.t_) && (v1.isBool_ == v2.isBool_) &&
         (v1.isAddressType() == v2.isAddressType());
}
bool operator!=(const MultiplierValue &v1, const MultiplierValue &v2) {
  return !(v1 == v2);
}
// --- End MultiplierValue Implementation ---

// --- GPU State Definition ---
class GPUState : public AbstractState<MultiplierValue, GPUState> {
public:
  GPUState() : numThreads_(MultiplierValue(TOP)) {}
  ~GPUState() override = default;

  GPUState(const GPUState &other)
      : AbstractState<MultiplierValue,
                      GPUState>() { // Explicit copy constructor
    valueMap_ = other.valueMap_;
    numThreads_ = other.numThreads_;
  }

  GPUState &operator=(const GPUState &other) { // Explicit copy assignment
    if (this != &other) {
      valueMap_ = other.valueMap_;
      numThreads_ = other.numThreads_;
    }
    return *this;
  }

  bool operator==(const AbstractState &other) const override {
    const GPUState &otherGPUState = static_cast<const GPUState &>(other);
    return (valueMap_ == otherGPUState.valueMap_) &&
           (numThreads_ == otherGPUState.numThreads_);
  }

  // Override operator= (needed to fulfill pure virtual)
  void operator=(const AbstractState &other) override {
    const GPUState &otherGPUState = static_cast<const GPUState &>(other);
    if (this != &otherGPUState) {
      valueMap_ = otherGPUState.valueMap_;
      numThreads_ = otherGPUState.numThreads_;
    }
  }

  GPUState mergeState(const GPUState &st) const override {
    GPUState result = *this;
    for (auto const &map_entry : st.valueMap_) {
      const Value *val = map_entry.first;
      const MultiplierValue &st_abstract_val = map_entry.second;
      if (result.hasValue(val)) {
        result.setValue(val, result.getValue(val).join(st_abstract_val));
      } else {
        result.setValue(val, st_abstract_val);
      }
    }
    result.numThreads_ = numThreads_.join(st.numThreads_);
    return result;
  }

  MultiplierValue getValue(const Value *in) const override {
    if (!in)
      return MultiplierValue(BOT);
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(in)) {
      if (CI->isZero())
        return MultiplierValue(ZERO);
      if (CI->isOne())
        return MultiplierValue(ONE);
      if (CI->isMinusOne())
        return MultiplierValue(NEGONE);
      return MultiplierValue(TOP);
    } else if (isa<ConstantPointerNull>(in)) {
      return MultiplierValue(ZERO);
    }
    auto it = valueMap_.find(in);
    if (it != valueMap_.end()) {
      return it->second;
    } else {
      return MultiplierValue(BOT);
    }
  }

  void setNumThreads(MultiplierValue v) { numThreads_ = v; }
  MultiplierValue getNumThreads() const { return numThreads_; }
  std::string getString() const override;

private:
  MultiplierValue numThreads_;
};

std::string GPUState::getString() const {
  std::string s = AbstractState::getString();
  s.append(" #t:").append(numThreads_.getString());
  return s;
}
// --- End GPU State ---

// --- The LLVM Pass for Barrier Divergence ---
namespace { // Start anonymous namespace

struct CUDADivergencePass : public FunctionPass {
  static char ID; // Declaration
  std::set<const Instruction *> DivergentBarriers;
  std::map<const BasicBlock *, GPUState> ConvergedBBExitStates;

  CUDADivergencePass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  // --- ExecuteInstruction (Simplified for divergence) ---
  GPUState ExecuteInstruction(const Instruction *I, GPUState st_in,
                              DominatorTree *DT) {
    GPUState st_out = st_in;
    if (!I)
      return st_out;

    if (isa<BinaryOperator>(I)) {
      const BinaryOperator *BO = cast<BinaryOperator>(I);
      MultiplierValue v1 = st_in.getValue(BO->getOperand(0));
      MultiplierValue v2 = st_in.getValue(BO->getOperand(1));
      MultiplierValue v;
      switch (BO->getOpcode()) {
      case Instruction::Add:
        v = v1 + v2;
        break;
      case Instruction::Sub:
        v = v1 + (-v2);
        break;
      case Instruction::Mul:
        v = v1 * v2;
        break;
      case Instruction::And:
        v = v1 && v2;
        break;
      case Instruction::Or:
        v = v1 || v2;
        break;
      default:
        v = MultiplierValue(TOP);
        break;
      }
      st_out.setValue(BO, v);
    } else if (isa<CastInst>(I)) {
      MultiplierValue v_in = st_in.getValue(I->getOperand(0));
      st_out.setValue(I, v_in);
      if (v_in.isAddressType() && I->getType()->isPointerTy())
        st_out.getValue(I).setAddressType();
      else
        st_out.getValue(I).setValueType();
    } else if (isa<CallInst>(I)) {
      const CallInst *CI = cast<CallInst>(I);
      Function *calledF = CI->getCalledFunction();
      if (calledF && calledF->hasName()) {
        StringRef name = calledF->getName();
        if (name == "llvm.nvvm.read.ptx.sreg.tid.x") {
          st_out.setValue(CI, MultiplierValue(ONE));
        } else if (name.startswith("llvm.nvvm.read.ptx.sreg.") &&
                   name != "llvm.nvvm.read.ptx.sreg.tid.x") {
          st_out.setValue(CI, MultiplierValue(ZERO));
        } else if (!name.startswith("llvm.nvvm.barrier") &&
                   !CI->getType()->isVoidTy()) {
          st_out.setValue(CI, MultiplierValue(TOP));
        }
      } else if (!CI->getType()->isVoidTy()) {
        st_out.setValue(CI, MultiplierValue(TOP));
      }
    } else if (isa<AllocaInst>(I)) {
      MultiplierValue pointerVal(ZERO, false);
      pointerVal.setAddressType();
      st_out.setValue(I, pointerVal);
    } else if (isa<LoadInst>(I)) {
      MultiplierValue v_ptr = st_in.getValue(I->getOperand(0));
      MultiplierValue v_loaded = (v_ptr.getType() == BOT)
                                     ? MultiplierValue(BOT)
                                     : MultiplierValue(TOP);
      if (I->getType()->isPointerTy()) {
        v_loaded.setAddressType();
      } else {
        v_loaded.setValueType();
      }
      st_out.setValue(I, v_loaded);
    } else if (isa<GetElementPtrInst>(I)) {
      MultiplierValue v_base = st_in.getValue(I->getOperand(0));
      MultiplierValue v_result = MultiplierValue(TOP);
      if (I->getNumOperands() == 2) {
        MultiplierValue v_idx = st_in.getValue(I->getOperand(1));
        v_result = v_base + v_idx;
      }
      if (v_base.isAddressType()) {
        v_result.setAddressType();
      }
      st_out.setValue(I, v_result);
    } else if (isa<PHINode>(I)) {
      const PHINode *PHI = cast<PHINode>(I);
      MultiplierValue v_phi = MultiplierValue(BOT);
      for (unsigned i = 0; i < PHI->getNumIncomingValues(); i++) {
        v_phi = v_phi.join(st_in.getValue(PHI->getIncomingValue(i)));
      }
      if (v_phi.getType() == BOT)
        v_phi = MultiplierValue(TOP);
      st_out.setValue(PHI, v_phi);
      // Inside GPUState ExecuteInstruction(const Instruction* I, ...)
    } else if (isa<CmpInst>(I)) {
      const CmpInst *CI = cast<CmpInst>(I);
      MultiplierValue v1 = st_in.getValue(CI->getOperand(0));
      MultiplierValue v2 = st_in.getValue(CI->getOperand(1));
      MultiplierValue v_cmp =
          MultiplierValue(TOP); // Default to TOP (Mixed/Unknown)

      // Only evaluate if both operands are known constants (ZERO, ONE, NEGONE)
      // or specific tid vs constant cases
      if (v1.getType() != TOP && v1.getType() != BOT && v2.getType() != TOP &&
          v2.getType() != BOT) {
        bool types_are_uniform = (v1.getType() == ZERO || v1.getType() == ONE ||
                                  v1.getType() == NEGONE) &&
                                 (v2.getType() == ZERO || v2.getType() == ONE ||
                                  v2.getType() == NEGONE);

        // Case 1: Both operands represent known uniform constants (0, 1, or -1)
        if (types_are_uniform &&
            !(v1.getType() == ONE || v1.getType() == NEGONE) &&
            !(v2.getType() == ONE || v2.getType() == NEGONE)) {
          // If both are ZERO
          int val1 = v1.getIntValue(); // Should be 0
          int val2 = v2.getIntValue(); // Should be 0
          bool result = false;
          switch (CI->getPredicate()) {
          // Signed comparisons
          case CmpInst::ICMP_EQ:
            result = (val1 == val2);
            break;
          case CmpInst::ICMP_NE:
            result = (val1 != val2);
            break;
          case CmpInst::ICMP_SLT:
            result = (val1 < val2);
            break;
          case CmpInst::ICMP_SGT:
            result = (val1 > val2);
            break;
          case CmpInst::ICMP_SLE:
            result = (val1 <= val2);
            break;
          case CmpInst::ICMP_SGE:
            result = (val1 >= val2);
            break;
          // Add unsigned later if needed (ULT, UGT, ULE, UGE)
          default:
            types_are_uniform = false;
            break; // Unhandled uniform predicate -> TOP
          }
          if (types_are_uniform) {
            // Map uniform result: True -> ONE, False -> ZERO (represents T/F
            // for V_bool)
            v_cmp = MultiplierValue(result ? ONE : ZERO, true);
          }
        }
        // Case 2: Check specific predicate helpers (EQ, NE already handled)
        else {
          switch (CI->getPredicate()) {
          case CmpInst::ICMP_EQ:
            v_cmp = eq(v1, v2);
            break;
          case CmpInst::ICMP_NE:
            v_cmp = neq(v1, v2);
            break;
          // Add simplified sgt/slt for tid vs 0 if needed
          // Example: sgt(tid, 0) -> should be mixed -> TOP
          // Example: slt(tid, N/2) -> should be mixed -> TOP
          // Defaulting others to TOP is safest if abstract ops aren't defined
          default:
            v_cmp = MultiplierValue(TOP);
            break;
          }
        }
        // If still TOP after specific checks, ensure it remains TOP
        if (v_cmp.getType() == BOT)
          v_cmp = MultiplierValue(TOP); // Avoid BOT if possible

      } else { // One or both operands are BOT or TOP
        v_cmp = (v1.getType() == BOT || v2.getType() == BOT)
                    ? MultiplierValue(BOT)
                    : MultiplierValue(TOP);
      }

      v_cmp.isBool_ = true;
      st_out.setValue(CI, v_cmp);

    } // End CmpInst Handling
    return st_out;
  }

  // --- Dataflow Analysis Driver (Execute) ---
  void Execute(Function &F, DominatorTree *DT,
               std::map<const BasicBlock *, GPUState>
                   &BBExitStates) // Only need exit states map
  {

    std::queue<const BasicBlock *> worklist;
    std::map<const BasicBlock *, GPUState> currentBBEntryStates;

    BasicBlock &entry = F.getEntryBlock();
    GPUState initialState = BuildInitialState(F);
    currentBBEntryStates[&entry] = initialState;
    worklist.push(&entry);

    int iterations = 0;
    const int MAX_ITERATIONS = F.size() * 20;

    std::set<const BasicBlock *> everInWorklist;
    everInWorklist.insert(&entry);

    while (!worklist.empty() && iterations < MAX_ITERATIONS) {
      const BasicBlock *BB = worklist.front();
      worklist.pop();
      iterations++;

      // Get/Compute state at entry by joining predecessors' exit states
      GPUState st_entry = GPUState(); // Default BOT state
      if (BB == &entry) {
        st_entry = currentBBEntryStates.at(BB);
      } else {
        for (const BasicBlock *Pred : predecessors(BB)) {
          if (BBExitStates.count(Pred)) { // If predecessor has been processed
            st_entry = st_entry.mergeState(BBExitStates.at(Pred));
          }
        }
      }

      // Check if entry state changed
      bool stateChanged = false;
      if (!currentBBEntryStates.count(BB) ||
          currentBBEntryStates.at(BB) != st_entry) {
        stateChanged = true;
        currentBBEntryStates[BB] = st_entry; // Update entry state
      }

      // If state didn't change at entry, no need to reprocess
      if (!stateChanged && iterations > 1 &&
          BBExitStates.count(BB)) { // Check if exit state already exists
        continue;
      }

      // Process instructions in block
      GPUState st_exit = st_entry;
      for (const Instruction &I : *BB) {
        st_exit = ExecuteInstruction(&I, st_exit, DT);
      }

      // Check if exit state changed & update successors
      if (!BBExitStates.count(BB) || BBExitStates.at(BB) != st_exit) {
        BBExitStates[BB] = st_exit;
        const Instruction *Term = BB->getTerminator(); // Corrected type
        for (unsigned i = 0; i < Term->getNumSuccessors();
             ++i) { // Corrected variable
          BasicBlock *Succ = Term->getSuccessor(i);
          // Add successor to worklist unconditionally if exit state changed
          // The entry state check at the top handles convergence.
          if (everInWorklist.find(Succ) == everInWorklist.end()) {
            worklist.push(Succ);
            everInWorklist.insert(Succ);
          } else {
            worklist.push(Succ); // Re-queue needed for convergence
          }
        }
      }
    } // End while

    if (iterations >= MAX_ITERATIONS) {
      outs() << "Dataflow analysis iteration limit\n";
    }
  }

  // --- BuildInitialState ---
  GPUState BuildInitialState(Function &F) const {
    GPUState st;
    st.setNumThreads(MultiplierValue(TOP));
    for (Function::const_arg_iterator argIt = F.arg_begin();
         argIt != F.arg_end(); ++argIt) {
      const Value *arg = &*argIt;
      MultiplierValue v(ZERO);
      if (arg->getType()->isPointerTy()) {
        v.setAddressType();
      }
      st.setValue(arg, v);
    }
    return st;
  }

  // --- runOnFunction ---
  bool runOnFunction(Function &F) override {
    outs() << "[ Algorithm ]Processing Function for Barrier Divergence: "
           << F.getName() << "\n";
    bool isKernel = false;
    // --- Kernel Identification Logic ---
    CallingConv::ID cc = F.getCallingConv();
    if (cc == CallingConv::PTX_Kernel)
      isKernel = true;
    if (!isKernel) {
      Module *M = F.getParent();
      if (M) {
        NamedMDNode *NMD = M->getNamedMetadata("nvvm.annotations");
        if (NMD) {
          for (unsigned i = 0, e = NMD->getNumOperands(); i < e; ++i) {
            MDNode *A = NMD->getOperand(i);
            if (A && A->getNumOperands() >= 3) {
              if (ValueAsMetadata *VMD =
                      dyn_cast<ValueAsMetadata>(A->getOperand(0).get())) {
                if (Function *AF = dyn_cast<Function>(VMD->getValue())) {
                  if (AF == &F) {
                    if (MDString *T =
                            dyn_cast<MDString>(A->getOperand(1).get())) {
                      if (T->getString() == "kernel") {
                        isKernel = true;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      // --- End Kernel Identification ---

      if (!isKernel) {
        return false;
      }

      DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      ConvergedBBExitStates.clear();
      std::map<const BasicBlock *, GPUState>
          convergedBBEntryStates; // Declared but not used after Execute
      Execute(F, DT, ConvergedBBExitStates);

      // --- Barrier Divergence Check ---
      DivergentBarriers.clear();

      for (Instruction &I : instructions(F)) {
        CallInst *BarrierInst = dyn_cast<CallInst>(&I);
        if (!BarrierInst)
          continue;
        Function *calledFunc = BarrierInst->getCalledFunction();
        if (!(calledFunc && calledFunc->isIntrinsic() &&
              (calledFunc->getName() == "llvm.nvvm.barrier0")))
          continue;

        BasicBlock *BB_Barrier = BarrierInst->getParent();
        DomTreeNode *BarrierNode = DT->getNode(BB_Barrier);
        if (!BarrierNode) {
          continue;
        }
        DomTreeNode *IDomNode = BarrierNode->getIDom();
        if (!IDomNode) {

          continue;
        }
        BasicBlock *DomBB = IDomNode->getBlock();
        if (!ConvergedBBExitStates.count(DomBB)) {

          continue;
        }

        const Instruction *Term = DomBB->getTerminator(); // Corrected type
        const BranchInst *CondBranch = dyn_cast<BranchInst>(Term);

        if (CondBranch && CondBranch->isConditional()) {
          if (CondBranch->getSuccessor(0) != BB_Barrier &&
              CondBranch->getSuccessor(1) != BB_Barrier) {

            continue; // Skip to next barrier check
          }
          // *** END ADDED CHECK ***
          Value *Cond = CondBranch->getCondition();

          const GPUState &stateAtDomExit = ConvergedBBExitStates.at(DomBB);
          MultiplierValue mvCond = stateAtDomExit.getValue(Cond);

          // Apply Rule: Divergent if condition is TOP, ONE, or NEGONE
          if (mvCond.getType() == TOP || mvCond.getType() == ONE ||
              mvCond.getType() == NEGONE) {

            DivergentBarriers.insert(BarrierInst);
          } else { // Condition is ZERO or BOT
          }
        } else {
        }
      }

      // Report results
      if (!DivergentBarriers.empty()) {
        outs() << "[ Algorithm ] Divergent Barrier Report for " << F.getName()
               << ":\n";
        for (const Instruction *barrier : DivergentBarriers) {
          unsigned line = 0;
          if (const DebugLoc &DL = barrier->getDebugLoc()) {
            line = DL.getLine();
          }
          outs() << "[RESULT]   Potential Divergent Barrier Detected at line "
                 << line << ":\n";
          barrier->print(outs());
          outs() << "\n";
        }
      } else {
        outs() << "[RESULT]   No divergent barriers detected for "
               << F.getName() << ".\n";
      }
      outs() << "[ Algorithm ] Finished analysis for: " << F.getName()
             << "\n\n";
      return false;
    }
    return false; // No changes made
  }

}; // end anonymous namespace

// --- Pass Registration ---
// Define ID outside the class and outside the anonymous namespace
char CUDADivergencePass::ID = 0;
} // namespace

static RegisterPass<CUDADivergencePass>
    X("cuda-divergence-check", // Registration name
      "CUDA Barrier Divergence Detector (Conceptual V8)",
      false, // Does not modify CFG
      true); // Analysis Pass
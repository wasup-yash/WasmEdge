#include "ast/component/instance.h"
#include "ast/module.h"
#include "common/errcode.h"
#include "common/types.h"
#include "executor/executor.h"

#include "runtime/instance/module.h"
#include "spdlog/spdlog.h"

#include <sstream>
#include <string_view>
#include <variant>

namespace WasmEdge {
namespace Executor {

using namespace std::literals;
using namespace AST::Component;
using namespace Runtime;

namespace {
Expect<void> pushType(Runtime::Instance::ComponentInstance &Comp,
                      std::vector<ValType> &Types, const ValueType &T) {
  // notice that we might need to convert one type to multiple types, and hence,
  // we must let this function control the vector need to be modified.
  if (std::holds_alternative<PrimValType>(T)) {
    switch (std::get<PrimValType>(T)) {
    case PrimValType::Bool:
    case PrimValType::Char:
    case PrimValType::S8:
      Types.push_back(ValType(TypeCode::I8));
      break;
    case PrimValType::U8:
      Types.push_back(ValType(TypeCode::U8));
      break;
    case PrimValType::S16:
    case PrimValType::U16:
      Types.push_back(ValType(TypeCode::I16));
      break;
    case PrimValType::S32:
    case PrimValType::U32:
      Types.push_back(ValType(TypeCode::I32));
      break;
    case PrimValType::S64:
    case PrimValType::U64:
      Types.push_back(ValType(TypeCode::I64));
      break;
    case PrimValType::Float32:
      Types.push_back(ValType(TypeCode::F32));
      break;
    case PrimValType::Float64:
      Types.push_back(ValType(TypeCode::F64));
      break;
    case PrimValType::String:
      Types.push_back(InterfaceType(TypeCode::String));
      break;
    }
  } else {
    auto Idx = std::get<TypeIndex>(T);
    auto RTy = Comp.getType(Idx);
    if (!RTy) {
      return Unexpect(RTy);
    }
    spdlog::warn("Type {} is not handled yet"sv, *RTy);
  }

  return {};
}

Expect<AST::FunctionType> convert(Runtime::Instance::ComponentInstance &Comp,
                                  const FuncType &DT) {
  Expect<void> Res;

  std::vector<ValType> ParamTypes{};
  for (const auto &P : DT.getParamList()) {
    Res = pushType(Comp, ParamTypes, P.getValType());
    if (!Res) {
      return Unexpect(Res);
    }
  }

  std::vector<ValType> ResultTypes{};
  if (std::holds_alternative<ValueType>(DT.getResultList())) {
    Res = pushType(Comp, ResultTypes, std::get<ValueType>(DT.getResultList()));
    if (!Res) {
      return Unexpect(Res);
    }
  } else {
    const auto &RL = DT.getResultList();
    for (const auto &R : std::get<std::vector<LabelValType>>(RL)) {
      Res = pushType(Comp, ResultTypes, R.getValType());
      if (!Res) {
        return Unexpect(Res);
      }
    }
  }

  return AST::FunctionType(ParamTypes, ResultTypes);
}

void flattenType(std::vector<WasmEdge::ValType> &Output, const ValType &Ty) {
  switch (Ty.getCode()) {
  case TypeCode::String:
    Output.push_back(TypeCode::I32);
    Output.push_back(TypeCode::I32);
    break;
  case TypeCode::List: {
    // TODO:
    // if maybe_length is not None:
    //     return flatten_type(elem_type) * maybe_length
    Output.push_back(TypeCode::I32);
    Output.push_back(TypeCode::I32);
    break;
  }
  case TypeCode::Record: {
    auto R = static_cast<const InterfaceType &>(Ty);
    for (auto FT : R.getArgs()) {
      flattenType(Output, FT);
    }
    break;
  }
  default:
    Output.push_back(Ty);
    break;
  }
}

} // namespace

class LiftTrans : public WasmEdge::Runtime::Component::HostFunctionBase {
public:
  LiftTrans(Executor *Exec, const AST::Component::FuncType &DefinedType,
            Instance::FunctionInstance *Func, Instance::MemoryInstance *M,
            Instance::FunctionInstance *R,
            Runtime::Instance::ComponentInstance &Comp)
      : HostFunctionBase(), Exec(Exec), LowerFunc(Func), Memory(M), Realloc(R) {
    // The convert is simply let component type to internal type.
    auto RFuncType = convert(Comp, DefinedType);
    if (!RFuncType) {
      spdlog::error("failed to lift function");
    }
    FuncType = *RFuncType;
    spdlog::info("lifted: {}"sv, FuncType);
  }

  Expect<void> run(Span<const ValInterface> Args,
                   Span<ValInterface> Rets) override {
    const auto &HigherFuncType = FuncType;

    uint32_t PI = 0;
    std::vector<ValVariant> LowerArgs{};
    for (auto &ParamTy : HigherFuncType.getParamTypes()) {
      switch (ParamTy.getCode()) {
      case TypeCode::String: {
        std::string_view Str = std::get<std::string>(Args[PI++]);

        auto StrSize = static_cast<uint32_t>(Str.size());
        std::vector<ValVariant> ReallocArgs{ValVariant(0), ValVariant(0),
                                            ValVariant(0), ValVariant(StrSize)};
        auto RPtr = Exec->invoke(Realloc, ReallocArgs,
                                 Realloc->getFuncType().getParamTypes());
        if (!RPtr) {
          return Unexpect(RPtr);
        }
        ValVariant PtrInMem = (*RPtr)[0].first;

        Memory->setBytes(std::vector<Byte>{Str.begin(), Str.end()},
                         PtrInMem.get<uint32_t>(), 0,
                         static_cast<uint32_t>(Str.size()));
        LowerArgs.push_back(PtrInMem);
        LowerArgs.push_back(StrSize);
        break;
      }
      default: {
        // usual type has no need conversion
        const ValVariant &Arg = std::get<ValVariant>(Args[PI++]);
        LowerArgs.push_back(Arg);
        break;
      }
      }
    }

    auto &LowerFuncType = LowerFunc->getFuncType();
    auto Res =
        Exec->invoke(LowerFunc, LowerArgs, LowerFuncType.getParamTypes());
    if (!Res) {
      return Unexpect(Res);
    }

    uint32_t RI = 0;
    uint32_t TakeI = 0;
    auto ResultList = *Res;
    for (auto const &HighTy : HigherFuncType.getReturnTypes()) {
      switch (HighTy.getCode()) {
      case TypeCode::String: {
        auto Idx = ResultList[TakeI++].first.get<uint32_t>();
        auto Size = ResultList[TakeI++].first.get<uint32_t>();
        auto Str = Memory->getStringView(Idx, Size);
        Rets[RI++].emplace<std::string>(std::string(Str.begin(), Str.end()));
        break;
      }
      default: {
        Rets[RI++].emplace<ValVariant>(ResultList[TakeI++].first);
        break;
      }
      }
    }

    return {};
  }

private:
  Executor *Exec;
  Instance::FunctionInstance *LowerFunc;
  Instance::MemoryInstance *Memory;
  Instance::FunctionInstance *Realloc;
};

std::unique_ptr<Instance::Component::FunctionInstance>
Executor::lifting(Runtime::Instance::ComponentInstance &Comp,
                  const FuncType &FuncType, Instance::FunctionInstance *Func,
                  Instance::MemoryInstance *Memory,
                  Instance::FunctionInstance *Realloc) {
  return std::make_unique<Instance::Component::FunctionInstance>(
      std::make_unique<LiftTrans>(this, FuncType, Func, Memory, Realloc, Comp));
}

class LowerTrans : public HostFunctionBase {
public:
  LowerTrans(Executor *Exec, Instance::Component::FunctionInstance *Func,
             Instance::MemoryInstance *Memory,
             Instance::FunctionInstance *Realloc)
      : HostFunctionBase(0), Exec(Exec), HigherFunc(Func), Memory(Memory),
        Realloc(Realloc) {
    auto &HigherType = HigherFunc->getFuncType();
    flattenFunctype(HigherType);
  }

  Expect<void> run(const Runtime::CallingFrame &, Span<const ValVariant> Args,
                   Span<ValVariant> Rets) override {
    auto &HigherFuncType = HigherFunc->getFuncType();

    uint32_t PI = 0;
    std::vector<ValInterface> HigherArgs{};
    for (auto &ParamTy : HigherFuncType.getParamTypes()) {
      switch (ParamTy.getCode()) {
      case TypeCode::String: {
        auto Idx = Args[PI++];
        auto Len = Args[PI++];
        std::string_view V =
            Memory->getStringView(Idx.get<uint32_t>(), Len.get<uint32_t>());
        ValInterface VI{};
        VI.emplace<std::string>(std::string(V.begin(), V.end()));
        HigherArgs.push_back(VI);
        break;
      }
      default:
        auto Arg = Args[PI++];
        HigherArgs.push_back(Arg);
        break;
      }
    }

    auto Res =
        Exec->invoke(HigherFunc, HigherArgs, HigherFuncType.getParamTypes());
    if (!Res) {
      return Unexpect(Res);
    }

    uint32_t RI = 0;
    for (auto &[RetVal, RetTy] : *Res) {
      switch (RetTy.getCode()) {
      case TypeCode::String: {
        auto const &Str = std::get<std::string>(RetVal);

        auto StrSize = static_cast<uint32_t>(Str.size());
        std::vector<ValVariant> ReallocArgs{ValVariant(0), ValVariant(0),
                                            ValVariant(0), ValVariant(StrSize)};
        auto RPtr = Exec->invoke(Realloc, ReallocArgs,
                                 Realloc->getFuncType().getParamTypes());
        if (!RPtr) {
          return Unexpect(RPtr);
        }
        ValVariant V = (*RPtr)[0].first;

        Memory->setBytes(std::vector<Byte>{Str.begin(), Str.end()},
                         V.get<uint32_t>(), 0,
                         static_cast<uint32_t>(Str.size()));
        Rets[RI++] = V;
        Rets[RI++] = ValVariant(StrSize);
        break;
      }
      default:
        Rets[RI++] = std::get<ValVariant>(RetVal);
        break;
      }
    }

    return {};
  }

private:
  void flattenFunctype(const AST::FunctionType &HigherType) {
    auto &FuncType = DefType.getCompositeType().getFuncType();
    for (auto &ParamTy : HigherType.getParamTypes()) {
      flattenType(FuncType.getParamTypes(), ParamTy);
    }
    for (auto &ReturnTy : HigherType.getReturnTypes()) {
      flattenType(FuncType.getReturnTypes(), ReturnTy);
    }

    spdlog::info("lower: {}"sv, FuncType);
  }

  Executor *Exec;
  /* HigherFunc: a component function we are wrapping
   */
  Instance::Component::FunctionInstance *HigherFunc;
  /* Memory: the shared memory from a certain core module defined in component
   */
  Instance::MemoryInstance *Memory;
  Instance::FunctionInstance *Realloc;
};

std::unique_ptr<Instance::FunctionInstance>
Executor::lowering(Instance::Component::FunctionInstance *Func,
                   Instance::MemoryInstance *Memory,
                   Instance::FunctionInstance *Realloc) {
  return std::make_unique<Instance::FunctionInstance>(
      std::make_unique<LowerTrans>(this, Func, Memory, Realloc));
}

class ResourceDropHostFunction
    : public WasmEdge::Runtime::Component::HostFunctionBase {
public:
  ResourceDropHostFunction(Executor *E, uint32_t Idx,
                           AST::Component::ResourceType &RT,
                           Runtime::Instance::ComponentInstance &C)
      : Exec{E}, TypIdx{Idx}, RTyp{RT}, Comp{C} {
    std::vector<ValType> ParamTypes{ValType(TypeCode::I32)};
    // NOTE: resource destructor only use type `i32`
    // 1. at sync mode: [i32] -> []
    // 2. at async mode: [i32] -> [i32]
    // for now, we ignore async case, to simplify our program here, but at
    // future shall make a full concept supportings.
    std::vector<ValType> ResultTypes{};
    FuncType = AST::FunctionType(ParamTypes, ResultTypes);
  }

  Expect<void> run(Span<const ValInterface> Args,
                   Span<ValInterface> /* Rets */) {
    if (Args.size() != 1 || !std::holds_alternative<ValVariant>(Args[0])) {
      spdlog::info("bad argument");
      return Unexpect(ErrCode::Value::ResourceDropArgument);
    }

    auto ResourceIdx = std::get<ValVariant>(Args[0]).get<uint32_t>();

    std::shared_ptr<Runtime::Instance::ResourceHandle> Handle =
        Comp.removeResource(TypIdx, ResourceIdx);
    if (Handle->isOwn()) {
      // TODO: assert borrowScope is None
      // TODO: trap lendCount != 0

      // TODO: Comp is RTyp.impl
      if (true) {
        if (*RTyp.getDestructor()) {
          auto Idx = *RTyp.getDestructor();
          auto F = Comp.getFunctionInstance(Idx);
          auto Arg = ValInterface(ValVariant(Handle->getRep()));
          Exec->invoke(F, {Arg}, {ValType(TypeCode::I32)});
        }
      } else {
        if (*RTyp.getDestructor()) {
          // TODO
          // caller_opts = CanonicalOptions(sync = sync)
          // callee_opts = CanonicalOptions(sync = rt.dtor_sync, callback =
          // rt.dtor_callback)
          // ft = FuncType([U32Type()],[])
          // callee = partial(canon_lift, callee_opts, rt.impl, ft, rt.dtor)
          // flat_results = await canon_lower(caller_opts, ft, callee, task,
          // [h.rep])
        } else {
          // task.trap_if_on_the_stack(rt.impl)
        }
      }
    } else {
      // TODO: Handle.borrowScope.todo -= 1
    }

    return {};
  }

private:
  Executor *Exec;
  uint32_t TypIdx;
  AST::Component::ResourceType &RTyp;
  Runtime::Instance::ComponentInstance &Comp;
};

std::unique_ptr<Instance::Component::FunctionInstance>
Executor::resourceDrop(uint32_t TypIdx, AST::Component::ResourceType &RTyp,
                       Runtime::Instance::ComponentInstance &CompInst) {
  return std::make_unique<Instance::Component::FunctionInstance>(
      std::make_unique<ResourceDropHostFunction>(this, TypIdx, RTyp, CompInst));
}

class CanonOptionVisitor {
private:
  Executor &ThisExecutor;
  Runtime::Instance::ComponentInstance &CompInst;

public:
  CanonOptionVisitor(Executor &E, Runtime::Instance::ComponentInstance &CInst)
      : ThisExecutor{E}, CompInst{CInst} {}

  // lift wrap a core wasm function to a component function, with proper
  // modification about canonical ABI.
  Expect<void> operator()(const Lift &L) {
    const auto &Opts = L.getOptions();

    Runtime::Instance::MemoryInstance *Mem = nullptr;
    Runtime::Instance::FunctionInstance *ReallocFunc = nullptr;
    for (auto &Opt : Opts) {
      if (std::holds_alternative<StringEncoding>(Opt)) {
        switch (std::get<StringEncoding>(Opt)) {
        case StringEncoding::UTF8:
          spdlog::warn("ignore utf8");
          break;
        case StringEncoding::UTF16:
          spdlog::warn("ignore utf16");
          break;
        case StringEncoding::Latin1:
          spdlog::warn("ignore latin1");
          break;
        default:
          assumingUnreachable();
        }
      } else if (std::holds_alternative<Memory>(Opt)) {
        auto MemIdx = std::get<Memory>(Opt).getMemIndex();
        auto RMem = CompInst.getCoreMemoryInstance(MemIdx);
        if (!RMem) {
          return Unexpect(RMem);
        }
        Mem = *RMem;
      } else if (std::holds_alternative<Realloc>(Opt)) {
        auto Res = CompInst.getCoreFunctionInstance(
            std::get<Realloc>(Opt).getFuncIndex());
        if (!Res) {
          return Unexpect(Res);
        }
        ReallocFunc = *Res;
      } else if (std::holds_alternative<PostReturn>(Opt)) {
        spdlog::error(ErrInfo::InfoAST(ASTNodeAttr::Sec_Canon));
        return Unexpect(ErrCode::Value::InvalidCanonOption);
      }
    }

    auto AstFuncType = CompInst.getType(L.getFuncTypeIndex());
    if (!AstFuncType) {
      spdlog::error("cannot lift type"sv);
      return Unexpect(AstFuncType);
    }
    if (unlikely(!std::holds_alternative<FuncType>(*AstFuncType))) {
      // It doesn't make sense if one tries to lift an instance not a
      // function, so unlikely happen.
      spdlog::error("cannot lift a non-function"sv);
      spdlog::error(ErrInfo::InfoAST(ASTNodeAttr::Sec_Canon));
      return Unexpect(ErrCode::Value::InvalidCanonOption);
    }

    auto RFuncInst = CompInst.getCoreFunctionInstance(L.getCoreFuncIndex());
    if (!RFuncInst) {
      return Unexpect(RFuncInst);
    }
    auto FuncInst = *RFuncInst;
    CompInst.addFunctionInstance(
        ThisExecutor.lifting(CompInst, std::get<FuncType>(*AstFuncType),
                             FuncInst, Mem, ReallocFunc));

    return {};
  }

  // lower sends a component function to a core wasm function, with proper
  // modification about canonical ABI.
  Expect<void> operator()(const Lower &L) {
    Runtime::Instance::MemoryInstance *Mem = nullptr;
    Runtime::Instance::FunctionInstance *ReallocFunc = nullptr;

    const auto &Opts = L.getOptions();
    for (auto &Opt : Opts) {
      if (std::holds_alternative<StringEncoding>(Opt)) {
        switch (std::get<StringEncoding>(Opt)) {
        case StringEncoding::UTF8:
          spdlog::warn("ignore utf8");
          break;
        case StringEncoding::UTF16:
          spdlog::warn("ignore utf16");
          break;
        case StringEncoding::Latin1:
          spdlog::warn("ignore latin1");
          break;
        default:
          assumingUnreachable();
        }
      } else if (std::holds_alternative<Memory>(Opt)) {
        auto MemIdx = std::get<Memory>(Opt).getMemIndex();
        auto RMem = CompInst.getCoreMemoryInstance(MemIdx);
        if (!RMem) {
          return Unexpect(RMem);
        }
        Mem = *RMem;
      } else if (std::holds_alternative<Realloc>(Opt)) {
        auto Res = CompInst.getCoreFunctionInstance(
            std::get<Realloc>(Opt).getFuncIndex());
        if (!Res) {
          return Unexpect(Res);
        }
        ReallocFunc = *Res;
      } else if (std::holds_alternative<PostReturn>(Opt)) {
        spdlog::error(ErrInfo::InfoAST(ASTNodeAttr::Sec_Canon));
        return Unexpect(ErrCode::Value::InvalidCanonOption);
      }
    }

    auto *FuncInst = CompInst.getFunctionInstance(L.getFuncIndex());
    CompInst.addCoreFunctionInstance(
        ThisExecutor.lowering(FuncInst, Mem, ReallocFunc));

    return {};
  }

  Expect<void> operator()(const ResourceNew &RNew) {
    auto TypIdx = RNew.getTypeIndex();
    auto RTyp = CompInst.getType(TypIdx);
    if (!RTyp) {
      return Unexpect(RTyp);
    }
    if (!std::holds_alternative<ResourceType>(*RTyp)) {
      spdlog::error(
          "resource.new cannot instantiate a deftype that's not a resource.");
      return Unexpect(ErrCode::Value::InvalidCanonOption);
    }

    auto ResourceTyp = std::get<ResourceType>(*RTyp);
    spdlog::info("get {}", ResourceTyp);
    spdlog::warn("resource.new is not supported yet"sv);

    return {};
  }

  Expect<void> operator()(const ResourceDrop &RDrop) {
    auto TypIdx = RDrop.getTypeIndex();
    auto RTyp = CompInst.getType(TypIdx);
    if (!RTyp) {
      return Unexpect(RTyp);
    }
    if (!std::holds_alternative<ResourceType>(*RTyp)) {
      spdlog::error("resource.drop cannot instantiate a deftype that's not a "
                    "resource.");
      return Unexpect(ErrCode::Value::InvalidCanonOption);
    }

    auto ResourceTyp = std::get<ResourceType>(*RTyp);
    auto Drop = ThisExecutor.resourceDrop(TypIdx, ResourceTyp, CompInst);
    Instance::Component::FunctionInstance *F = Drop.get();
    CompInst.addCoreFunctionInstance(
        ThisExecutor.lowering(F, nullptr, nullptr));

    return {};
  }

  Expect<void> operator()(const ResourceRep &) {
    spdlog::warn("resource.rep is not supported yet"sv);
    return Unexpect(ErrCode::Value::InvalidCanonOption);
  }
};

Expect<void>
Executor::instantiate(Runtime::StoreManager &,
                      Runtime::Instance::ComponentInstance &CompInst,
                      const AST::Component::CanonSection &CanonSec) {
  for (const Canon &C : CanonSec.getContent()) {
    auto Res = std::visit(CanonOptionVisitor{*this, CompInst}, C);
    if (!Res) {
      return Unexpect(Res);
    }
  }

  return {};
}

} // namespace Executor
} // namespace WasmEdge

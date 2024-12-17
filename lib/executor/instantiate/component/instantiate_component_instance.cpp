#include "ast/component/instance.h"
#include "ast/module.h"
#include "common/errcode.h"
#include "executor/executor.h"

#include "runtime/instance/module.h"

#include <string_view>
#include <variant>

namespace WasmEdge {
namespace Executor {

using namespace std::literals;
using namespace AST::Component;

// In this function, we will create a new module instance and insert it into
// component module instance index space.
Expect<void>
Executor::instantiate(Runtime::StoreManager &StoreMgr,
                      Runtime::Instance::ComponentInstance &CompInst,
                      const AST::Component::CoreInstanceSection &Sec) {
  for (const CoreInstanceExpr &InstExpr : Sec.getContent()) {
    if (std::holds_alternative<CoreInstantiate>(InstExpr)) {
      auto &Instantiate = std::get<CoreInstantiate>(InstExpr);

      for (auto &Arg : Instantiate.getArgs()) {
        // instantiate a list of `(with (name $instance))`
        // each $instance get named as `name` as statement tell
        StoreMgr.addNamedModule(Arg.getName(),
                                CompInst.getModuleInstance(Arg.getIndex()));
      }
      const AST::Module &Mod = CompInst.getModule(Instantiate.getModuleIdx());
      auto Res = instantiate(StoreMgr, Mod);
      if (!Res) {
        return Unexpect(Res);
      }
      CompInst.addModuleInstance(std::move(*Res));
    } else {
      auto &InstantiateExpr =
          std::get<AST::Component::CoreInlineExports>(InstExpr);

      // create an immediate module instance, which has no name
      //
      // this happened usually at `(with xxx)` statement, where we can have a
      // module instance expression.
      auto M = std::make_unique<Runtime::Instance::ModuleInstance>("");

      for (auto &S : InstantiateExpr.getExports()) {
        const auto &SortIdx = S.getSortIdx();
        switch (SortIdx.getSort()) {
        case CoreSort::Func: {
          // The module instance takes functions and export them

          // get stored core function
          auto Res = CompInst.getCoreFunctionInstance(SortIdx.getSortIdx());
          if (!Res) {
            return Unexpect(Res);
          }
          M->exportFunction(S.getName(), *Res);
          break;
        }
        case CoreSort::Table: {
          auto Res = CompInst.getCoreTableInstance(SortIdx.getSortIdx());
          if (!Res) {
            return Unexpect(Res);
          }
          M->exportTable(S.getName(), *Res);
          break;
        }
        case CoreSort::Memory: {
          auto Res = CompInst.getCoreMemoryInstance(SortIdx.getSortIdx());
          if (!Res) {
            return Unexpect(Res);
          }
          M->exportMemory(S.getName(), *Res);
          break;
        }
        case CoreSort::Global: {
          auto Res = CompInst.getCoreGlobalInstance(SortIdx.getSortIdx());
          if (!Res) {
            return Unexpect(Res);
          }
          M->exportGlobal(S.getName(), *Res);
          break;
        }
        case CoreSort::Type:
        case CoreSort::Module:
        case CoreSort::Instance: {
          spdlog::error(
              "A module instance cannot exports types, modules, or instances"sv);
          return Unexpect(ErrCode::Value::CoreInvalidExport);
        }
        }
      }

      // insert this immediate module instance into the component instance
      CompInst.addModuleInstance(std::move(M));
    }
  }
  return {};
}

Expect<void>
Executor::instantiate(Runtime::StoreManager &StoreMgr,
                      Runtime::Instance::ComponentInstance &CompInst,
                      const AST::Component::InstanceSection &Sec) {
  for (auto &InstExpr : Sec.getContent()) {
    if (std::holds_alternative<AST::Component::Instantiate>(InstExpr)) {
      auto &Instantiate = std::get<AST::Component::Instantiate>(InstExpr);

      for (auto &Arg : Instantiate.getArgs()) {
        const auto &Idx = Arg.getIndex();
        const auto &S = Idx.getSort();
        if (std::holds_alternative<CoreSort>(S)) {
          // TODO: insert below into mapping
          switch (std::get<CoreSort>(S)) {
          case CoreSort::Func:
            spdlog::warn("incomplete (with {}) core:func"sv, Arg.getName());
            break;
          case CoreSort::Table:
            spdlog::warn("incomplete (with {}) core:table"sv, Arg.getName());
            break;
          case CoreSort::Memory:
            spdlog::warn("incomplete (with {}) core:memory"sv, Arg.getName());
            break;
          case CoreSort::Global:
            spdlog::warn("incomplete (with {}) core:global"sv, Arg.getName());
            break;
          case CoreSort::Type:
            spdlog::warn("incomplete (with {}) core:type"sv, Arg.getName());
            break;
          case CoreSort::Module:
            spdlog::warn("incomplete (with {}) core:module"sv, Arg.getName());
            break;
          case CoreSort::Instance:
            StoreMgr.addNamedModule(
                Arg.getName(), CompInst.getModuleInstance(Idx.getSortIdx()));
            break;
          }
        } else if (std::holds_alternative<SortCase>(S)) {
          switch (std::get<SortCase>(S)) {
          case SortCase::Func: // TODO: figure out how to do this registry
            spdlog::warn("incomplete (with {}) function"sv, Arg.getName());
            break;
          case SortCase::Value: // TODO: figure out how to do this registry
            spdlog::warn("incomplete (with {}) value"sv, Arg.getName());
            break;
          case SortCase::Type: // TODO: figure out how to do this registry
            spdlog::warn("incomplete (with {}) type"sv, Arg.getName());
            break;
          case SortCase::Component:
            if (auto Res = StoreMgr.registerComponent(
                    Arg.getName(),
                    CompInst.getComponentInstance(Idx.getSortIdx()));
                !Res) {
              spdlog::error("failed to register component instance"sv);
              return Unexpect(Res);
            }
            break;
          case SortCase::Instance:
            // TODO: figure out how to do this registry
            spdlog::warn("incomplete (with {}) instance"sv, Arg.getName());
            break;
          }
        }
      }
      auto &C = CompInst.getComponent(Instantiate.getComponentIdx());
      auto Res = instantiate(StoreMgr, C);
      if (!Res) {
        return Unexpect(Res);
      }
      auto Inst = std::move(*Res);
      CompInst.addComponentInstance(std::move(Inst));
    } else {
      std::get<CompInlineExports>(InstExpr).getExports();
      // TODO: complete inline exports
      spdlog::warn("incomplete component inline exports"sv);
    }
  }
  return {};
}

} // namespace Executor
} // namespace WasmEdge
